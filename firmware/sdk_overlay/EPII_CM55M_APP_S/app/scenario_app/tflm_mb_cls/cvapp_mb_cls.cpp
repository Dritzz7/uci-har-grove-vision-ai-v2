#include <cstdio>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp_mb_cls.h"
#include "cisdp_sensor.h"

#include "WE2_core.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif
#include "img_proc_helium.h"


#include "xprintf.h"
#include "spi_master_protocol.h"
#include "cisdp_cfg.h"
#include "memory_manage.h"
#include "har_mpu6050.h"
#include <send_result.h>
#include <forward_list>

#define INPUT_IMAGE_CHANNELS 3


#define MB_CLS_INPUT_TENSOR_WIDTH   224
#define MB_CLS_INPUT_TENSOR_HEIGHT  224
#define MB_CLS_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS


#define MB_CLS_DBG_APP_LOG 0


// #define EACH_STEP_TICK
#define TOTAL_STEP_TICK

uint32_t systick_1, systick_2;
uint32_t loop_cnt_1, loop_cnt_2;
#define CPU_CLK	0xffffff+1
static uint32_t capture_image_tick = 0;
#ifdef TRUSTZONE_SEC
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE	BASE_ADDR_APB_U55_CTRL
#endif
#endif


using namespace std;

namespace {

constexpr int tensor_arena_size = 450*1024;

static uint32_t tensor_arena=0;

struct ethosu_driver ethosu_drv; /* Default Ethos-U device driver */
tflite::MicroInterpreter *mb_cls_int_ptr=nullptr;
TfLiteTensor *mb_cls_input, *mb_cls_output;
};

static void xprintf_confidence_percent(float score)
{
    int scaled = (int)(score * 10000.0f + (score >= 0.0f ? 0.5f : -0.5f));

    if (scaled < 0) {
        scaled = 0;
    }

    int whole = scaled / 100;
    int frac = scaled % 100;
    xprintf("%d.%d%d%%", whole, (frac / 10) % 10, frac % 10);
}

static void _arm_npu_irq_handler(void)
{
    /* Call the default interrupt handler from the NPU driver */
    ethosu_irq_handler(&ethosu_drv);
}

/**
 * @brief  Initialises the NPU IRQ
 **/
static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;

    /* Register the EthosU IRQ handler in our vector table.
     * Note, this handler comes from the EthosU driver */
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);

    /* Enable the IRQ */
    NVIC_EnableIRQ(ethosu_irqnum);

}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;

    /* Initialise the IRQ */
    _arm_npu_irq_init();

    /* Initialise Ethos-U55 device */
#if TFLM2209_U55TAG2205
	const void * ethosu_base_address = (void *)(U55_BASE);
#else
	void * const ethosu_base_address = (void *)(U55_BASE);
#endif

    if (0 != (err = ethosu_init(
                            &ethosu_drv,             /* Ethos-U driver device pointer */
                            ethosu_base_address,     /* Ethos-U NPU's base address. */
                            NULL,       /* Pointer to fast mem area - NULL for U55. */
                            0, /* Fast mem region size. */
							security_enable,                       /* Security enable. */
							privilege_enable))) {                   /* Privilege enable. */
	xprintf("failed to initalise Ethos-U device\n");
            return err;
        }

    xprintf("Ethos-U55 device initialised\n");

    return 0;
}


int cv_mb_cls_init(bool security_enable, bool privilege_enable, uint32_t model_addr) {
	int ercode = 0;

	//set memory allocation to tensor_arena
	tensor_arena = mm_reserve_align(tensor_arena_size,0x20); //1mb
	xprintf("TA[%x]\r\n",tensor_arena);


	if(_arm_npu_init(security_enable, privilege_enable)!=0)
		return -1;

	if(model_addr != 0) {
		static const tflite::Model*mb_cls_model = tflite::GetModel((const void *)model_addr);

		if (mb_cls_model->version() != TFLITE_SCHEMA_VERSION) {
			xprintf(
				"[ERROR] mb_cls_model's schema version %d is not equal "
				"to supported version %d\n",
				mb_cls_model->version(), TFLITE_SCHEMA_VERSION);
			return -1;
		}
		else {
			xprintf("mb_cls model's schema version %d\n", mb_cls_model->version());
		}
        static tflite::MicroMutableOpResolver<4> mb_cls_op_resolver;

        if (kTfLiteOk != mb_cls_op_resolver.AddEthosU())
            xprintf("AddEthosU FAIL\n");

        if (kTfLiteOk != mb_cls_op_resolver.AddStridedSlice())
            xprintf("AddStridedSlice FAIL\n");

        if (kTfLiteOk != mb_cls_op_resolver.AddPack())
            xprintf("AddPack FAIL\n");

        if (kTfLiteOk != mb_cls_op_resolver.AddReshape())
            xprintf("AddReshape FAIL\n");

		static tflite::MicroInterpreter mb_cls_static_interpreter(mb_cls_model, mb_cls_op_resolver,
				(uint8_t*)tensor_arena, tensor_arena_size);



		if(mb_cls_static_interpreter.AllocateTensors()!= kTfLiteOk) {
			return false;
		}
		mb_cls_int_ptr = &mb_cls_static_interpreter;
		mb_cls_input = mb_cls_static_interpreter.input(0);
		mb_cls_output = mb_cls_static_interpreter.output(0);
		xprintf("MODEL INFO\n");

	xprintf("Input bytes : %d\n", mb_cls_input->bytes);

	xprintf("Input dims : ");
	for (int i = 0; i < mb_cls_input->dims->size; i++) {
		xprintf("%d ", mb_cls_input->dims->data[i]);
	}
	xprintf("\n");

	xprintf("Output bytes : %d\n", mb_cls_output->bytes);

	xprintf("Output dims : ");
	for (int i = 0; i < mb_cls_output->dims->size; i++) {
		xprintf("%d ", mb_cls_output->dims->data[i]);
	}
	xprintf("\n");

		if (mb_cls_input->dims->size != 3 ||
			mb_cls_input->dims->data[0] != 1 ||
			mb_cls_input->dims->data[1] != (int)HAR_WINDOW_SAMPLES ||
			mb_cls_input->dims->data[2] != (int)HAR_INPUT_CHANNELS ||
			mb_cls_input->type != kTfLiteInt8) {
			xprintf("Unexpected HAR input tensor; expected int8 [1,128,9]\n");
			return -1;
		}
		if (har_mpu6050_init() != 0) {
			xprintf("HW-123/MPU6050 initialisation failed\n");
			return -1;
		}
	}

	xprintf("initial done\n");
	return ercode;
}


int cv_mb_cls_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob)
{
    if(mb_cls_int_ptr == nullptr) return -1;

		if (har_mpu6050_fill_window(mb_cls_input->data.int8,
		                            mb_cls_input->bytes,
		                            mb_cls_input->params.scale,
		                            mb_cls_input->params.zero_point) != 0) {
			xprintf("Failed to acquire HAR IMU window\r\n");
			return -1;
		}

    TfLiteStatus invoke_status = mb_cls_int_ptr->Invoke();
    if(invoke_status != kTfLiteOk)
    {
        xprintf("Invoke failed");
        return -1;
    }

    const char *har_labels[]={
        "Walking",
        "Walking Upstairs",
        "Walking Downstairs",
        "Sitting",
        "Standing",
        "Laying"
    };

    float scale=((TfLiteAffineQuantization*)mb_cls_output->quantization.params)->scale->data[0];
    int zp=((TfLiteAffineQuantization*)mb_cls_output->quantization.params)->zero_point->data[0];

    int best=0;
    float best_score=-1e9;


    for(int i=0;i<6;i++)
    {
        float score=((float)mb_cls_output->data.int8[i]-zp)*scale;
        if(score>best_score){
            best_score=score;
            best=i;
        }
    }

    xprintf("Prediction : %s (confidence ", har_labels[best]);
    xprintf_confidence_percent(best_score);
    xprintf(")\r\n");

    algoresult_yolov8n_ob->obr[0].class_idx=best;
    algoresult_yolov8n_ob->obr[0].confidence=best_score;

    return 0;
}

int cv_mb_cls_deinit()
{

	return 0;
}
