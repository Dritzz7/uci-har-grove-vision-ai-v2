# UCI-HAR on Grove Vision AI V2

Six-class human activity recognition using the UCI-HAR dataset, a 1D CNN,
INT8 quantization, Arm Vela, the Ethos-U55 NPU, and a live HW-123/MPU6050 IMU.

The deployed classes are:

1. Walking
2. Walking upstairs
3. Walking downstairs
4. Sitting
5. Standing
6. Laying

## Pipeline

```text
UCI-HAR inertial windows [128, 9]
        -> two Conv1D + MaxPool blocks
        -> Keras FP32 model
        -> full-integer TFLite INT8 model
        -> Vela compilation for Ethos-U55-64
        -> Grove Vision AI V2 firmware
        -> live MPU6050 sampling, filtering and inference
```

Each 2.56-second input window contains 128 samples at 50 Hz with nine channels:

```text
body_acc_x, body_acc_y, body_acc_z,
body_gyro_x, body_gyro_y, body_gyro_z,
total_acc_x, total_acc_y, total_acc_z
```

The live firmware uses a 64-sample stride, giving 50% window overlap.

## Results

| Item | Result |
|---|---:|
| Keras test accuracy | 92.23% |
| INT8 TFLite test accuracy | 91.58% |
| Trainable parameters | 35,302 |
| Neural-network MACs | 227,532 |
| Vela peak SRAM estimate | 5.078 KiB |
| Vela inference-time estimate | 0.0913 ms |
| Vela accelerator | Ethos-U55-64 |

The Vela timing is a compiler estimate for the selected system configuration,
not an end-to-end measurement including IMU window acquisition.

The network consists of two convolutional blocks:

```text
Input [128,9]
  -> Conv1D(16, kernel=5) -> MaxPool(2)
  -> Conv1D(32, kernel=3) -> MaxPool(2)
  -> Flatten -> Dense(32) -> Dropout(0.2)
  -> Dense(6, softmax)
```

## Repository contents

```text
scripts/
  03_train_UCI-HAR.ipynb       training, evaluation and INT8 conversion
  har_dashboard.py             local browser GUI for live serial predictions
model/
  uci_har_model_fp32.keras     trained Keras model
  uci_har_model_int8.tflite    fully quantized model
  uci_har_label_mapping.json
  uci_har_training_history.png
vela_out_u55_64/
  uci_har_model_int8_vela.tflite
  ...summary...csv             Vela performance and memory report
firmware/
  sdk_overlay/                 MPU6050 and inference integration sources
  prebuilt/output.img          packaged Grove Vision AI V2 firmware
  install_into_seeed_sdk.sh
```

The UCI dataset, virtual environment, complete Seeed SDK, build directories and
large NumPy test exports are intentionally excluded.

## Dataset

Download **Human Activity Recognition Using Smartphones** from:

<https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>

Extract it to:

```text
dataset/UCI HAR Dataset/
```

The dataset contains recordings from 30 volunteers using a waist-mounted
Samsung Galaxy S II. Accelerometer and gyroscope signals were sampled at 50 Hz.

## Python environment

All project commands below are intended for Ubuntu/WSL.

```bash
cd /path/to/uci-har-grove-vision-ai-v2

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the notebook:

```bash
jupyter notebook scripts/03_train_UCI-HAR.ipynb
```

## Vela compilation

The final deployment model was compiled for Ethos-U55 with 64 MACs/cycle:

```bash
vela model/uci_har_model_int8.tflite \
  --accelerator-config=ethos-u55-64 \
  --system-config=Ethos_U55_High_End_Embedded \
  --memory-mode=Shared_Sram \
  --output-dir=vela_out_u55_64
```

## Hardware

- Grove Vision AI Module V2
- HW-123 breakout with MPU6050
- Grove-to-header wires
- USB cable for power, UART and flashing

### IMU wiring

| Grove Vision AI V2 | HW-123/MPU6050 |
|---|---|
| Yellow, SCL | SCL |
| White, SDA | SDA |
| Red, VCC | VCC |
| Black, GND | GND |
| GND | AD0 |

Leave INT, XDA and XCL disconnected. Grounding AD0 selects I2C address
`0x68`, which is the address used by the firmware.

### Sensor mounting

Mount the sensor rigidly at the front-left waist:

- breadboard/sensor plane vertical;
- natural top edge toward the head;
- component side facing outward, away from the body;
- back side against the belt or clothing.

The firmware maps the measured axes into the model frame as:

```text
Model X = Native Y
Model Y = -Native X
Model Z = Native Z
```

When upright and still, the expected result is approximately:

```text
IMU_NATIVE_ACC_mg X=0 Y=+1000 Z=0
IMU_MODEL_ACC_mg  X=+1000 Y=0 Z=0
```

## Firmware integration

The overlay targets this Seeed SDK revision:

```text
Repository: https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2
Commit:     d3265e20d75fe20faccbf185d24454ad08b2fda8
```

Clone the SDK and install the overlay:

```bash
git clone --recursive \
  https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2.git

git -C Seeed_Grove_Vision_AI_Module_V2 checkout \
  d3265e20d75fe20faccbf185d24454ad08b2fda8

./firmware/install_into_seeed_sdk.sh \
  ./Seeed_Grove_Vision_AI_Module_V2
```

Build and package using the Arm GNU 13.2.Rel1 toolchain:

```bash
cd Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S
make clean
make -j2

cd ../we2_image_gen_local
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf \
  input_case1_secboot/
./we2_local_image_gen project_case1_blp_wlcsp.json
```

## Flashing from WSL

Close any serial monitor first. From the Seeed SDK directory:

```bash
python xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="/path/to/uci-har-grove-vision-ai-v2/vela_out_u55_64/uci_har_model_int8_vela.tflite 0xB7B000 0x00000"
```

Reset the board when the flashing tool requests it.

Open the serial monitor:

```bash
python -m serial.tools.miniterm /dev/ttyACM0 921600
```

Exit miniterm with `Ctrl+]`.

## Browser dashboard

The dashboard replaces the visible serial terminal while the board continues
performing inference on the NPU.

```bash
python scripts/har_dashboard.py \
  --serial-port=/dev/ttyACM0 \
  --baudrate=921600
```

Then open <http://127.0.0.1:8765>.

Use `python scripts/har_dashboard.py --demo` to test it without hardware.

## Important limitations

- A motionless board on a table is outside the training distribution. Because
  there is no unknown class, the model must still select one of six activities.
- Sitting and standing overlap strongly in UCI-HAR. Both usually have model
  X near +1 g and are separated by smaller orientation and motion differences.
- Mounting position and axis orientation must remain consistent.
- The online 0.3 Hz gravity filter approximates the UCI-HAR preprocessing, but
  a different IMU and body placement still introduce domain shift.

## Third-party data and software

- UCI-HAR is distributed by the UCI Machine Learning Repository under CC BY
  4.0. The downloaded dataset is not included here.
- The firmware overlay is based on the Seeed/Himax Grove Vision AI Module V2
  SDK. Its MIT license is preserved in `LICENSES/SEEED_SDK_MIT.txt`.
- TensorFlow, Arm Vela and their dependencies retain their own licenses.
