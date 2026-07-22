#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT="d3265e20d75fe20faccbf185d24454ad08b2fda8"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/Seeed_Grove_Vision_AI_Module_V2" >&2
    exit 2
fi

SDK_ROOT="$(realpath "$1")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_ROOT="$SCRIPT_DIR/sdk_overlay"
APP_REL="EPII_CM55M_APP_S/app/scenario_app/tflm_mb_cls"
APP_DIR="$SDK_ROOT/$APP_REL"
MAKEFILE="$SDK_ROOT/EPII_CM55M_APP_S/makefile"

if [[ ! -d "$APP_DIR" || ! -f "$MAKEFILE" ]]; then
    echo "Not a compatible Seeed Grove Vision AI Module V2 SDK: $SDK_ROOT" >&2
    exit 1
fi

if git -C "$SDK_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ACTUAL_COMMIT="$(git -C "$SDK_ROOT" rev-parse HEAD)"
    if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
        echo "Warning: overlay was tested with SDK commit $EXPECTED_COMMIT" >&2
        echo "Current SDK commit is $ACTUAL_COMMIT" >&2
    fi
fi

for file in har_mpu6050.c har_mpu6050.h cvapp_mb_cls.cpp tflm_mb_cls.c; do
    cp "$OVERLAY_ROOT/$APP_REL/$file" "$APP_DIR/$file"
done

if grep -q '^APP_TYPE = ' "$MAKEFILE"; then
    sed -i 's/^APP_TYPE = .*/APP_TYPE = tflm_mb_cls/' "$MAKEFILE"
else
    echo "Could not find APP_TYPE in $MAKEFILE" >&2
    exit 1
fi

echo "Installed the UCI-HAR MPU6050 firmware overlay."
echo
echo "Next:"
echo "  cd $SDK_ROOT/EPII_CM55M_APP_S"
echo "  make clean"
echo "  make -j2"
