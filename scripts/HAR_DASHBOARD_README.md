# HAR browser dashboard

This dashboard reads Grove Vision AI V2 prediction messages from USB serial and
shows the current activity, confidence, connection status, and a rolling
confidence graph at `http://127.0.0.1:8765`.

The serial connection is still the transport between the board and laptop. The
dashboard replaces the visible serial terminal; it does not change the firmware
or move inference away from the Ethos-U55.

## Start in demo mode

Run this first to check the interface without hardware:

```bash
cd /path/to/uci-har-grove-vision-ai-v2

.venv/bin/python \
scripts/har_dashboard.py \
--demo
```

Open this address in a browser:

```text
http://127.0.0.1:8765
```

Stop the dashboard with `Ctrl+C`.

## Start with the Grove Vision AI V2

Attach the USB device to WSL and verify that `/dev/ttyACM0` exists. Close
miniterm or any other program using the serial device, then run:

```bash
cd /path/to/uci-har-grove-vision-ai-v2

.venv/bin/python \
scripts/har_dashboard.py \
--serial-port=/dev/ttyACM0 \
--baudrate=921600
```

Open:

```text
http://127.0.0.1:8765
```

If PySerial is missing from the WSL virtual environment:

```bash
.venv/bin/python \
-m pip install pyserial==3.5
```

## Expected firmware message

The parser accepts variants such as:

```text
Prediction : Standing confidence: 0.93
Prediction: Walking (confidence 87%)
Activity=Walking Upstairs score=0.814
```

If the activity changes but the dashboard says `Not reported`, ensure the
firmware prints the confidence on the same line as the activity prediction.

## Troubleshooting

- `Permission denied`: ensure the WSL user belongs to `dialout`, then restart
  the WSL session.
- `Device or resource busy`: close miniterm and other serial applications.
- `No such file`: reattach the USB device to WSL and confirm the device name.
- Dashboard opens but remains on `Waiting`: press the board reset button and
  watch the connection indicator.
