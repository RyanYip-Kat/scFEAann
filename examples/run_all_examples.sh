#!/usr/bin/env bash
#
# Run every scFEAann example in dependency order.
#
# Defaults match the original analysis setting: all cells and 100 epochs.
# The optional per-cell gradient calculation is disabled by default because it
# refits every cell; use --with-gradient to include it.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

INPUT_PATH="$PROJECT_ROOT/testdata/testdata.h5ad"
OUTPUT_DIR="$PROJECT_ROOT/outputs/examples_full_100ep"
MAX_CELLS=0
EPOCHS=100
DEVICE="auto"
CLASSES="all"
BENCHMARK_REPEATS=3
RUN_GRADIENT=0
SKIP_BENCHMARK=0
RESUME=0
DRY_RUN=0
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

usage() {
    cat <<'EOF'
Usage: examples/run_all_examples.sh [options]

Run the global scFEA model, all supermodule models, the SCI report, and the
performance benchmark. Defaults use all cells and 100 training epochs.

Options:
  --input PATH                 Input H5AD (default: testdata/testdata.h5ad)
  --output-dir PATH            Output directory
  --max-cells N                0 uses all cells; positive values sample (default: 0)
  --epochs N                   Training epochs (default: 100)
  --device DEVICE              Torch device (default: auto)
  --classes LIST               all or comma-separated supermodule selectors
  --benchmark-repeats N        Timed inference repeats (default: 3)
  --with-gradient              Add the expensive per-cell gradient example
  --skip-benchmark             Do not run scfeaann_benchmark.py
  --resume                     Skip stages whose validated output already exists
  --dry-run                    Print commands without creating files or fitting
  -h, --help                   Show this help

Environment:
  PYTHON_BIN                   Python executable (default: .venv/bin/python)
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

require_value() {
    local option="$1"
    local value="${2-}"
    if [[ -z "$value" ]]; then
        die "$option requires a value"
    fi
}

while (($#)); do
    case "$1" in
        --input)
            require_value "$1" "${2-}"
            INPUT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            require_value "$1" "${2-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-cells)
            require_value "$1" "${2-}"
            MAX_CELLS="$2"
            shift 2
            ;;
        --epochs)
            require_value "$1" "${2-}"
            EPOCHS="$2"
            shift 2
            ;;
        --device)
            require_value "$1" "${2-}"
            DEVICE="$2"
            shift 2
            ;;
        --classes)
            require_value "$1" "${2-}"
            CLASSES="$2"
            shift 2
            ;;
        --benchmark-repeats)
            require_value "$1" "${2-}"
            BENCHMARK_REPEATS="$2"
            shift 2
            ;;
        --with-gradient)
            RUN_GRADIENT=1
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=1
            shift
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help)"
            ;;
    esac
done

[[ "$MAX_CELLS" =~ ^[0-9]+$ ]] || die "max-cells must be zero or greater"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "epochs must be greater than zero"
[[ "$BENCHMARK_REPEATS" =~ ^[1-9][0-9]*$ ]] \
    || die "benchmark-repeats must be greater than zero"
[[ -n "$CLASSES" ]] || die "classes must not be empty"

if [[ "$INPUT_PATH" != /* ]]; then
    INPUT_PATH="$PROJECT_ROOT/$INPUT_PATH"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi
if [[ "$PYTHON_BIN" != /* ]]; then
    PYTHON_BIN="$PROJECT_ROOT/$PYTHON_BIN"
fi

[[ -f "$INPUT_PATH" ]] || die "input H5AD does not exist: $INPUT_PATH"
[[ -x "$PYTHON_BIN" ]] || die "Python executable is not available: $PYTHON_BIN"
for example in \
    scfeaann_testdata_flow.py \
    scfeaann_supermodule_flow.py \
    scfeaann_report.py \
    scfeaann_benchmark.py
do
    [[ -f "$SCRIPT_DIR/$example" ]] || die "example is missing: $example"
done

GLOBAL_OUTPUT="$OUTPUT_DIR/global_scfeaann.h5ad"
SUPERMODULE_OUTPUT="$OUTPUT_DIR/supermodule_scfeaann.h5ad"
REPORT_DIR="$OUTPUT_DIR/report"
REPORT_MANIFEST="$REPORT_DIR/report_manifest.json"
BENCHMARK_OUTPUT="$OUTPUT_DIR/benchmark.json"
LOG_DIR="$OUTPUT_DIR/logs"
SUMMARY_FILE="$OUTPUT_DIR/run_summary.tsv"
ENVIRONMENT_FILE="$OUTPUT_DIR/environment.txt"

GLOBAL_CMD=(
    "$PYTHON_BIN" "$SCRIPT_DIR/scfeaann_testdata_flow.py"
    --input "$INPUT_PATH"
    --output "$GLOBAL_OUTPUT"
    --max-cells "$MAX_CELLS"
    --epochs "$EPOCHS"
    --device "$DEVICE"
    --quiet
)
if ((RUN_GRADIENT)); then
    GLOBAL_CMD+=(--gradient)
fi

SUPERMODULE_CMD=(
    "$PYTHON_BIN" "$SCRIPT_DIR/scfeaann_supermodule_flow.py"
    --input "$GLOBAL_OUTPUT"
    --output "$SUPERMODULE_OUTPUT"
    --classes "$CLASSES"
    --group-key dataset_cluster
    --epochs "$EPOCHS"
    --device "$DEVICE"
    --quiet
    --stop-on-error
)

CLASSES_COMPACT="${CLASSES//[[:space:]]/}"
if [[ "${CLASSES_COMPACT,,}" == "all" ]]; then
    REPORT_INPUT="$SUPERMODULE_OUTPUT"
else
    # Partial class runs intentionally leave other class slots as NaN.
    # The report therefore uses the global result, which has complete axes.
    REPORT_INPUT="$GLOBAL_OUTPUT"
fi

REPORT_CMD=(
    "$PYTHON_BIN" "$SCRIPT_DIR/scfeaann_report.py"
    --input "$REPORT_INPUT"
    --output-dir "$REPORT_DIR"
    --dataset-key dataset
    --cluster-key cluster
    --embedding-method umap
    --random-state 16
    --dpi 300
)

BENCHMARK_CMD=(
    "$PYTHON_BIN" "$SCRIPT_DIR/scfeaann_benchmark.py"
    --input "$INPUT_PATH"
    --max-cells "$MAX_CELLS"
    --epochs "$EPOCHS"
    --device "$DEVICE"
    --prediction-batch-sizes "1,64,1024"
    --repeats "$BENCHMARK_REPEATS"
    --output-json "$BENCHMARK_OUTPUT"
    --quiet
)

print_command() {
    local stage="$1"
    shift
    printf 'DRY-RUN [%s]' "$stage"
    printf ' %q' "$@"
    printf '\n'
}

if ((DRY_RUN)); then
    printf 'INPUT=%q\n' "$INPUT_PATH"
    printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
    printf 'MAX_CELLS=%s\n' "$MAX_CELLS"
    printf 'EPOCHS=%s\n' "$EPOCHS"
    printf 'DEVICE=%s\n' "$DEVICE"
    printf 'CLASSES=%s\n' "$CLASSES"
    print_command "global" "${GLOBAL_CMD[@]}"
    print_command "supermodules" "${SUPERMODULE_CMD[@]}"
    print_command "report" "${REPORT_CMD[@]}"
    if ((SKIP_BENCHMARK == 0)); then
        print_command "benchmark" "${BENCHMARK_CMD[@]}"
    fi
    exit 0
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$REPORT_DIR"
export MPLCONFIGDIR="$OUTPUT_DIR/.matplotlib"
export XDG_CACHE_HOME="$OUTPUT_DIR/.cache"
export NUMBA_CACHE_DIR="$OUTPUT_DIR/.numba"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$NUMBA_CACHE_DIR"

if ((RESUME == 0)) || [[ ! -f "$SUMMARY_FILE" ]]; then
    printf 'stage\tstatus\tstarted_at\tfinished_at\tseconds\tartifact\n' \
        >"$SUMMARY_FILE"
fi

{
    printf 'started_at=%s\n' "$(date -Iseconds)"
    printf 'project_root=%s\n' "$PROJECT_ROOT"
    printf 'input=%s\n' "$INPUT_PATH"
    printf 'output_dir=%s\n' "$OUTPUT_DIR"
    printf 'max_cells=%s\n' "$MAX_CELLS"
    printf 'epochs=%s\n' "$EPOCHS"
    printf 'device=%s\n' "$DEVICE"
    printf 'classes=%s\n' "$CLASSES"
    printf 'with_gradient=%s\n' "$RUN_GRADIENT"
    printf 'skip_benchmark=%s\n' "$SKIP_BENCHMARK"
    printf 'python_bin=%s\n' "$PYTHON_BIN"
    printf 'mplconfigdir=%s\n' "$MPLCONFIGDIR"
    printf 'xdg_cache_home=%s\n' "$XDG_CACHE_HOME"
    printf 'numba_cache_dir=%s\n' "$NUMBA_CACHE_DIR"
    "$PYTHON_BIN" --version
} >"$ENVIRONMENT_FILE"

append_summary() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >>"$SUMMARY_FILE"
}

artifact_ready() {
    local artifact="$1"
    [[ -s "$artifact" ]]
}

run_stage() {
    local stage="$1"
    local artifact="$2"
    shift 2

    if ((RESUME)) && artifact_ready "$artifact"; then
        printf 'SKIP [%s]: artifact already exists: %s\n' "$stage" "$artifact"
        append_summary "$stage" "skipped" "$(date -Iseconds)" \
            "$(date -Iseconds)" "0" "$artifact"
        return 0
    fi

    local started_at finished_at start_epoch finish_epoch elapsed status
    local stage_log="$LOG_DIR/$stage.log"
    started_at="$(date -Iseconds)"
    start_epoch="$(date +%s)"
    printf 'START [%s]\n' "$stage"
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'

    set +e
    "$@" 2>&1 | tee "$stage_log"
    status=${PIPESTATUS[0]}
    set -e

    finished_at="$(date -Iseconds)"
    finish_epoch="$(date +%s)"
    elapsed=$((finish_epoch - start_epoch))
    if ((status != 0)); then
        append_summary "$stage" "failed" "$started_at" "$finished_at" \
            "$elapsed" "$artifact"
        printf 'FAILED [%s]: exit=%s; log=%s\n' \
            "$stage" "$status" "$stage_log" >&2
        exit "$status"
    fi
    if ! artifact_ready "$artifact"; then
        append_summary "$stage" "missing-artifact" "$started_at" \
            "$finished_at" "$elapsed" "$artifact"
        printf 'FAILED [%s]: expected artifact is missing or empty: %s\n' \
            "$stage" "$artifact" >&2
        exit 1
    fi

    append_summary "$stage" "success" "$started_at" "$finished_at" \
        "$elapsed" "$artifact"
    printf 'DONE [%s] in %ss\n' "$stage" "$elapsed"
}

printf 'Starting scFEAann examples: %s cells, %s epochs, device=%s\n' \
    "$([[ "$MAX_CELLS" == 0 ]] && printf all || printf '%s' "$MAX_CELLS")" \
    "$EPOCHS" "$DEVICE"
printf 'Output directory: %s\n' "$OUTPUT_DIR"

run_stage "global" "$GLOBAL_OUTPUT" "${GLOBAL_CMD[@]}"
run_stage "supermodules" "$SUPERMODULE_OUTPUT" "${SUPERMODULE_CMD[@]}"
run_stage "report" "$REPORT_MANIFEST" "${REPORT_CMD[@]}"
if ((SKIP_BENCHMARK == 0)); then
    run_stage "benchmark" "$BENCHMARK_OUTPUT" "${BENCHMARK_CMD[@]}"
fi

printf 'All requested example stages completed successfully.\n'
printf 'Summary: %s\n' "$SUMMARY_FILE"
