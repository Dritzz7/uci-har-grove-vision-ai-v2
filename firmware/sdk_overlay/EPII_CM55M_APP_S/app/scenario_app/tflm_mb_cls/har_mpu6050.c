#include "har_mpu6050.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "hx_drv_iic.h"
#include "hx_drv_scu.h"
#include "timer_interface.h"
#include "xprintf.h"

#define MPU_ADDR                  0x68U
#define REG_SMPLRT_DIV            0x19U
#define REG_CONFIG                0x1AU
#define REG_GYRO_CONFIG           0x1BU
#define REG_ACCEL_CONFIG          0x1CU
#define REG_INT_ENABLE            0x38U
#define REG_INT_STATUS            0x3AU
#define REG_ACCEL_XOUT_H          0x3BU
#define REG_PWR_MGMT_1            0x6BU
#define REG_PWR_MGMT_2            0x6CU
#define REG_WHO_AM_I              0x75U
#define DATA_READY                0x01U
#define ACCEL_LSB_PER_G           16384.0f
#define GYRO_LSB_PER_DPS          131.0f
#define DEG_TO_RAD                0.01745329251994329577f
#define CALIBRATION_SAMPLES       200U
#define READY_TIMEOUT_MS          50U
#define RECONNECT_DELAY_MS        1000U
#define READ_FAILURES_BEFORE_REINIT 3U

typedef struct {
    float value[3];
    uint8_t next;
    bool seeded;
} median3_t;

typedef struct {
    float b0, b1, b2, a1, a2;
} sos_coeff_t;

typedef struct {
    float z1[2];
    float z2[2];
    bool seeded;
} sos_filter_t;

// Fungsi scipy.signal.butter(3, 20, fs=50) (Median Filter) //
static const sos_coeff_t k_noise_20hz[2] = {
    {0.52762438f, 0.52762438f, 0.0f, 0.50952545f, 0.0f},
    {1.0f, 2.0f, 1.0f, 1.25051643f, 0.54572332f},
};

// Fungsi scipy.signal.butter(3, 0.3, fs=50) (Gravity Filter)//
static const sos_coeff_t k_gravity_03hz[2] = {
    {0.00000645184913f, 0.0000129036983f, 0.00000645184913f,
     -0.962994051f, 0.0f},
    {1.0f, 1.0f, 0.0f, -1.96161218f, 0.963006955f},
};

static median3_t g_accel_median[3];
static median3_t g_gyro_median[3];
static sos_filter_t g_accel_noise[3];
static sos_filter_t g_gyro_noise[3];
static sos_filter_t g_gravity[3];
static float g_gyro_bias_raw[3];
static float g_window[HAR_WINDOW_SAMPLES][HAR_INPUT_CHANNELS];
static uint32_t g_write_index;
static uint32_t g_valid_samples;
static uint32_t g_consecutive_read_failures;
static bool g_ready;

static void i2c_error_cb(void *status)
{
    HX_DRV_DEV_IIC *obj = (HX_DRV_DEV_IIC *)status;
    xprintf("HAR I2C error: %d\r\n", obj->iic_info.err_state);
}

static int write_reg(uint8_t reg, uint8_t value)
{
    return (int)hx_drv_i2cm_write_data(USE_DW_IIC_0, MPU_ADDR,
                                       &reg, 1, &value, 1);
}

static int read_regs(uint8_t reg, uint8_t *data, uint8_t length)
{
    return (int)hx_drv_i2cm_write_restart_read(USE_DW_IIC_0, MPU_ADDR,
                                               &reg, 1, data, length);
}

static int16_t be_i16(const uint8_t *data)
{
    return (int16_t)(((uint16_t)data[0] << 8) | data[1]);
}

static int read_raw(int16_t accel[3], int16_t gyro[3])
{
    uint32_t elapsed;
    int read_status;
    uint8_t status = 0U;
    uint8_t raw[14];

    for (elapsed = 0; elapsed < READY_TIMEOUT_MS; ++elapsed) {
        read_status = read_regs(REG_INT_STATUS, &status, 1);
        if (read_status != 0) {
            xprintf("MPU6050 I2C read error\r\n");
            return -1;
        }
        if ((status & DATA_READY) != 0U) {
            break;
        }
        hx_drv_timer_cm55x_delay_ms(1, TIMER_STATE_DC);
    }
    if (elapsed == READY_TIMEOUT_MS ||
        read_regs(REG_ACCEL_XOUT_H, raw, sizeof(raw)) != 0) {
        xprintf("MPU6050 sample timeout/read error\r\n");
        return -1;
    }
    accel[0] = be_i16(&raw[0]);
    accel[1] = be_i16(&raw[2]);
    accel[2] = be_i16(&raw[4]);
    gyro[0] = be_i16(&raw[8]);
    gyro[1] = be_i16(&raw[10]);
    gyro[2] = be_i16(&raw[12]);
    return 0;
}

static float median3(median3_t *s, float input)
{
    float a, b, c, tmp;

    if (!s->seeded) {
        s->value[0] = s->value[1] = s->value[2] = input;
        s->next = 0U;
        s->seeded = true;
    }
    s->value[s->next] = input;
    s->next = (uint8_t)((s->next + 1U) % 3U);
    a = s->value[0];
    b = s->value[1];
    c = s->value[2];
    if (a > b) { tmp = a; a = b; b = tmp; }
    if (b > c) { tmp = b; b = c; c = tmp; }
    if (a > b) { tmp = a; a = b; b = tmp; }
    return b;
}

static void sos_seed(sos_filter_t *s, const sos_coeff_t c[2], float input)
{
    uint32_t i;
    float x = input;

    for (i = 0; i < 2U; ++i) {
        const float gain = (c[i].b0 + c[i].b1 + c[i].b2) /
                           (1.0f + c[i].a1 + c[i].a2);
        const float y = gain * x;
        s->z1[i] = y - c[i].b0 * x;
        s->z2[i] = c[i].b2 * x - c[i].a2 * y;
        x = y;
    }
    s->seeded = true;
}

static float sos(sos_filter_t *s, const sos_coeff_t c[2], float input)
{
    uint32_t i;
    float x = input;

    if (!s->seeded) {
        sos_seed(s, c, input);
    }
    for (i = 0; i < 2U; ++i) {
        const float y = c[i].b0 * x + s->z1[i];
        s->z1[i] = c[i].b1 * x - c[i].a1 * y + s->z2[i];
        s->z2[i] = c[i].b2 * x - c[i].a2 * y;
        x = y;
    }
    return x;
}

static void map_axes(const float sensor[3], float model[3])
{
    model[0] = (float)HAR_AXIS_X_SIGN * sensor[HAR_AXIS_X_SOURCE];
    model[1] = (float)HAR_AXIS_Y_SIGN * sensor[HAR_AXIS_Y_SOURCE];
    model[2] = (float)HAR_AXIS_Z_SIGN * sensor[HAR_AXIS_Z_SOURCE];
}

static int calibrate_gyro(void)
{
    uint32_t sample, axis;
    int16_t accel[3], gyro[3];
    float sum[3] = {0.0f, 0.0f, 0.0f};

    xprintf("Keep HW-123 still: gyro calibration (4 seconds)\r\n");
    for (sample = 0; sample < CALIBRATION_SAMPLES; ++sample) {
        if (read_raw(accel, gyro) != 0) {
            return -1;
        }
        for (axis = 0; axis < 3U; ++axis) {
            sum[axis] += (float)gyro[axis];
        }
    }
    for (axis = 0; axis < 3U; ++axis) {
        g_gyro_bias_raw[axis] = sum[axis] / (float)CALIBRATION_SAMPLES;
    }
    return 0;
}

static int processed_sample(float output[HAR_INPUT_CHANNELS])
{
    uint32_t axis;
    int16_t accel_raw[3], gyro_raw[3];
    float accel_sensor[3], gyro_sensor[3], accel[3], gyro[3];

    if (read_raw(accel_raw, gyro_raw) != 0) {
        return -1;
    }
    for (axis = 0; axis < 3U; ++axis) {
        accel_sensor[axis] = (float)accel_raw[axis] / ACCEL_LSB_PER_G;
        gyro_sensor[axis] = (((float)gyro_raw[axis] - g_gyro_bias_raw[axis]) /
                             GYRO_LSB_PER_DPS) * DEG_TO_RAD;
    }
    map_axes(accel_sensor, accel);
    map_axes(gyro_sensor, gyro);

#if HAR_AXIS_DEBUG_PRINT
    /*
     * This function runs at 50 Hz, so print one sample every second.
     * NATIVE identifies the MPU6050 board axes. MODEL shows the values
     * after the HAR_AXIS_*_SOURCE and HAR_AXIS_*_SIGN mapping.
     */
    static uint32_t axis_debug_sample_count = 0U;

    axis_debug_sample_count++;
    if (axis_debug_sample_count >= 50U) {
        axis_debug_sample_count = 0U;

        xprintf(
            "IMU_NATIVE_ACC_mg X=%d Y=%d Z=%d\r\n",
            (int)(accel_sensor[0] * 1000.0f),
            (int)(accel_sensor[1] * 1000.0f),
            (int)(accel_sensor[2] * 1000.0f)
        );

        xprintf(
            "IMU_MODEL_ACC_mg  X=%d Y=%d Z=%d\r\n",
            (int)(accel[0] * 1000.0f),
            (int)(accel[1] * 1000.0f),
            (int)(accel[2] * 1000.0f)
        );
    }
#endif

    for (axis = 0; axis < 3U; ++axis) {
        const float total_acc =
            sos(&g_accel_noise[axis], k_noise_20hz,
                median3(&g_accel_median[axis], accel[axis]));
        const float body_gyro =
            sos(&g_gyro_noise[axis], k_noise_20hz,
                median3(&g_gyro_median[axis], gyro[axis]));
        const float gravity_acc =
            sos(&g_gravity[axis], k_gravity_03hz, total_acc);
        output[axis] = total_acc - gravity_acc;
        output[3U + axis] = body_gyro;
        output[6U + axis] = total_acc;
    }
    return 0;
}

static int8_t quantize(float value, float scale, int32_t zero_point)
{
    const float scaled = value / scale;
    int32_t q = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    q += zero_point;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

int har_mpu6050_init(void)
{
    uint8_t who = 0U;

    memset(g_accel_median, 0, sizeof(g_accel_median));
    memset(g_gyro_median, 0, sizeof(g_gyro_median));
    memset(g_accel_noise, 0, sizeof(g_accel_noise));
    memset(g_gyro_noise, 0, sizeof(g_gyro_noise));
    memset(g_gravity, 0, sizeof(g_gravity));
    memset(g_window, 0, sizeof(g_window));
    g_write_index = g_valid_samples = 0U;
    g_consecutive_read_failures = 0U;
    g_ready = false;

    xprintf("IMU_STATUS: INITIALIZING - keep sensor still\r\n");

    hx_drv_scu_set_PA2_pinmux(SCU_PA2_PINMUX_I2C_M_SCL, 1);
    hx_drv_scu_set_PA3_pinmux(SCU_PA3_PINMUX_I2C_M_SDA, 1);
    hx_drv_i2cm_init(USE_DW_IIC_0, HX_I2C_HOST_MST_0_BASE,
                     DW_IIC_SPEED_FAST);
    hx_drv_i2cm_set_err_cb(USE_DW_IIC_0, i2c_error_cb);

    if (write_reg(REG_PWR_MGMT_1, 0x80U) != 0) return -1;
    hx_drv_timer_cm55x_delay_ms(100, TIMER_STATE_DC);
    if (write_reg(REG_PWR_MGMT_1, 0x01U) != 0 ||
        write_reg(REG_PWR_MGMT_2, 0x00U) != 0) return -1;
    hx_drv_timer_cm55x_delay_ms(20, TIMER_STATE_DC);

    if (read_regs(REG_WHO_AM_I, &who, 1) != 0 || who != 0x68U) {
        xprintf("MPU6050 WHO_AM_I=0x%x, expected 0x68\r\n", who);
        return -1;
    }
    // 1 kHz/(19+1)=50 Hz, +/-2 g, +/-250 dps, hardware DLPF 44/42 Hz. //
    if (write_reg(REG_SMPLRT_DIV, 19U) != 0 ||
        write_reg(REG_CONFIG, 3U) != 0 ||
        write_reg(REG_GYRO_CONFIG, 0U) != 0 ||
        write_reg(REG_ACCEL_CONFIG, 0U) != 0 ||
        write_reg(REG_INT_ENABLE, DATA_READY) != 0) return -1;
    hx_drv_timer_cm55x_delay_ms(50, TIMER_STATE_DC);
    if (calibrate_gyro() != 0) return -1;

    g_ready = true;
    xprintf("HW-123/MPU6050 ready at 0x68, 50 Hz\r\n");
    xprintf("IMU_STATUS: READY\r\n");
    return 0;
}

int har_mpu6050_fill_window(int8_t *tensor, size_t tensor_bytes,
                            float input_scale, int32_t input_zero_point)
{
    uint32_t collect, sample, channel;

    if (tensor == NULL || input_scale <= 0.0f ||
        tensor_bytes != HAR_WINDOW_SAMPLES * HAR_INPUT_CHANNELS) {
        return -1;
    }

    /*
     * A disconnected VCC jumper power-cycles the MPU6050 and clears all of
     * its configuration registers. Merely retrying a sample is therefore not
     * sufficient after the wire is reconnected. Reinitialise the I2C
     * controller and sensor here so recovery does not require a board reset
     * or reopening the serial dashboard.
     */
    if (!g_ready) {
        xprintf("IMU_STATUS: RECONNECTING - keep sensor still\r\n");
        hx_drv_timer_cm55x_delay_ms(RECONNECT_DELAY_MS, TIMER_STATE_DC);
        if (har_mpu6050_init() != 0) {
            xprintf("IMU_STATUS: DISCONNECTED - automatic retry pending\r\n");
            return -1;
        }
        xprintf("IMU_STATUS: RECOVERED\r\n");
    }

    collect = g_valid_samples < HAR_WINDOW_SAMPLES
                  ? HAR_WINDOW_SAMPLES - g_valid_samples
                  : HAR_WINDOW_STRIDE;

    for (sample = 0; sample < collect; ++sample) {
        if (processed_sample(g_window[g_write_index]) != 0) {
            ++g_consecutive_read_failures;
            if (g_consecutive_read_failures >= READ_FAILURES_BEFORE_REINIT) {
                g_ready = false;
                g_write_index = 0U;
                g_valid_samples = 0U;
                memset(g_window, 0, sizeof(g_window));
                xprintf(
                    "IMU_STATUS: DISCONNECTED - automatic recovery enabled\r\n"
                );
            }
            return -1;
        }
        g_consecutive_read_failures = 0U;
        g_write_index = (g_write_index + 1U) % HAR_WINDOW_SAMPLES;
        if (g_valid_samples < HAR_WINDOW_SAMPLES) ++g_valid_samples;
    }
    for (sample = 0; sample < HAR_WINDOW_SAMPLES; ++sample) {
        const uint32_t ring = (g_write_index + sample) % HAR_WINDOW_SAMPLES;
        for (channel = 0; channel < HAR_INPUT_CHANNELS; ++channel) {
            tensor[sample * HAR_INPUT_CHANNELS + channel] =
                quantize(g_window[ring][channel], input_scale,
                         input_zero_point);
        }
    }
    return 0;
}
