# Firmware integration

This directory contains only the project-specific integration, not the complete
980 MB Seeed/Himax SDK.

## Files

- `har_mpu6050.c`: MPU6050 I2C driver, calibration, filters, ring buffer and
  INT8 input quantization.
- `har_mpu6050.h`: window dimensions, axis mapping and debug configuration.
- `cvapp_mb_cls.cpp`: validates the `[1,128,9]` tensor, acquires an IMU
  window, invokes TensorFlow Lite Micro and prints the class/confidence.
- `tflm_mb_cls.c`: runs continuous HAR inference instead of the camera
  classification loop.
- `prebuilt/output.img`: latest packaged firmware image. Flash it together
  with the Vela model from the repository root.

## Signal preprocessing

The live input order matches the training notebook:

```text
[body_acc_xyz, body_gyro_xyz, total_acc_xyz]
```

Processing is performed at 50 Hz:

1. Read accelerometer and gyroscope data over I2C.
2. Apply axis remapping to the UCI-HAR model frame.
3. Apply a three-sample median filter.
4. Apply a 20 Hz third-order Butterworth noise filter.
5. Apply a 0.3 Hz third-order Butterworth low-pass filter for gravity.
6. Calculate `body_acc = total_acc - gravity_acc`.
7. Store 128 samples with a stride of 64.
8. Quantize using the input tensor's scale and zero point.

Gyroscope bias is calibrated for four seconds during startup. Keep the sensor
completely still until calibration finishes.

If an MPU6050 jumper is disconnected while the system is running, the firmware
marks the current window invalid and automatically retries sensor
initialisation. After reconnecting all wires, keep the sensor still while the
four-second gyroscope calibration runs again. A board reset or dashboard
restart is not required.

## Debug output

`HAR_AXIS_DEBUG_PRINT` in `har_mpu6050.h` controls the once-per-second raw
and mapped acceleration messages:

```c
#define HAR_AXIS_DEBUG_PRINT 1
```

Set it to `0` for a quieter final demonstration, then rebuild and package the
firmware.
