#ifndef HAR_MPU6050_H
#define HAR_MPU6050_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HAR_WINDOW_SAMPLES 128U
#define HAR_WINDOW_STRIDE   64U
#define HAR_INPUT_CHANNELS   9U

#ifndef HAR_AXIS_X_SOURCE
#define HAR_AXIS_X_SOURCE 1
#endif
#ifndef HAR_AXIS_Y_SOURCE
#define HAR_AXIS_Y_SOURCE 0
#endif
#ifndef HAR_AXIS_Z_SOURCE
#define HAR_AXIS_Z_SOURCE 2
#endif
#ifndef HAR_AXIS_X_SIGN
#define HAR_AXIS_X_SIGN 1
#endif
#ifndef HAR_AXIS_Y_SIGN
#define HAR_AXIS_Y_SIGN -1
#endif
#ifndef HAR_AXIS_Z_SIGN
#define HAR_AXIS_Z_SIGN 1
#endif

/*
 * Print native and model-mapped acceleration once per second.
 * Values use milli-g units: +1000 mg is +1 g.
 * Set to 0 after the final mounting orientation is determined.
 */
#define HAR_AXIS_DEBUG_PRINT 1

int har_mpu6050_init(void);
int har_mpu6050_fill_window(int8_t *tensor, size_t tensor_bytes,
                            float input_scale, int32_t input_zero_point);

#ifdef __cplusplus
}
#endif

#endif
