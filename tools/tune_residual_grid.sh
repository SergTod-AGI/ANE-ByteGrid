#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ./build/train ]]; then
  echo "build/train is missing; run 'make build/train' first." >&2
  exit 1
fi

results_path="${ANE_TUNE_RESULTS:-build/tune_results.tsv}"
mkdir -p "$(dirname "$results_path")"

locals=(${=ANE_TUNE_LOCALS:-"1.00 1.05 1.10"})
globals=(${=ANE_TUNE_GLOBALS:-"1.20 1.25 1.30"})
mlps=(${=ANE_TUNE_MLPS:-"0.30 0.35 0.40"})
depths=(${=ANE_TUNE_DEPTHS:-"0.10 0.15 0.20"})
head_weights=(${=ANE_TUNE_HEAD_WEIGHTS:-"1.00"})
head_logits=(${=ANE_TUNE_HEAD_LOGITS:-"1.00"})

warmup="${ANE_TUNE_WARMUP:-1}"
iters="${ANE_TUNE_ITERS:-3}"
log_stats="${ANE_TUNE_LOG_STATS:-1}"

tmp_log="$(mktemp /tmp/ane-tune.XXXXXX)"
trap 'rm -f "$tmp_log"' EXIT

echo -e "local\tglobal\tmlp\tdepth\thead_weight\thead_logit\tcross_entropy\tperplexity\tavg_ms\tcompile_ms\tload_ms\thidden_naninf\thidden_sat\thidden_var\toutput_naninf\toutput_sat\toutput_var\tstable\tcomposite\tstatus" > "$results_path"

for local_mul in "${locals[@]}"; do
  for global_mul in "${globals[@]}"; do
    for mlp_mul in "${mlps[@]}"; do
      for depth_power in "${depths[@]}"; do
        for head_weight in "${head_weights[@]}"; do
          for head_logit in "${head_logits[@]}"; do
        echo "running local=$local_mul global=$global_mul mlp=$mlp_mul depth=$depth_power head_weight=$head_weight head_logit=$head_logit" >&2
        if ! ANE_BG_ALPHA_LOCAL_MUL="$local_mul" \
             ANE_BG_ALPHA_GLOBAL_MUL="$global_mul" \
             ANE_BG_ALPHA_MLP_MUL="$mlp_mul" \
             ANE_BG_ALPHA_DEPTH_POWER="$depth_power" \
             ANE_BG_HEAD_WEIGHT_SCALE="$head_weight" \
             ANE_BG_HEAD_LOGIT_SCALE="$head_logit" \
             ANE_BG_LOG_SURFACE_STATS="$log_stats" \
             ANE_BG_WARMUP="$warmup" \
             ANE_BG_ITERS="$iters" \
             ./build/train > "$tmp_log" 2>&1; then
          echo -e "${local_mul}\t${global_mul}\t${mlp_mul}\t${depth_power}\t${head_weight}\t${head_logit}\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tNO\tinf\tfailed" >> "$results_path"
          continue
        fi

        cross_entropy="$(sed -n 's/.*cross_entropy=\([0-9.]*\).*/\1/p' "$tmp_log" | tail -n 1)"
        perplexity="$(sed -n 's/.*perplexity=\([0-9.]*\).*/\1/p' "$tmp_log" | tail -n 1)"
        avg_ms="$(sed -n 's/.*avg=\([0-9.]*\) ms\/pass.*/\1/p' "$tmp_log" | tail -n 1)"
        compile_ms="$(sed -n 's/.*compile=\([0-9.]*\) ms load=.*/\1/p' "$tmp_log" | tail -n 1)"
        load_ms="$(sed -n 's/.*load=\([0-9.]*\) ms.*/\1/p' "$tmp_log" | tail -n 1)"
        guardrails_parsed="$(awk '
          /guardrails: hidden_naninf=/ {
            for (i = 1; i <= NF; ++i) {
              split($i, kv, "=");
              if (kv[1] == "hidden_naninf") hidden_naninf = kv[2];
              if (kv[1] == "hidden_sat") hidden_sat = kv[2];
              if (kv[1] == "hidden_var") hidden_var = kv[2];
              if (kv[1] == "output_naninf") output_naninf = kv[2];
              if (kv[1] == "output_sat") output_sat = kv[2];
              if (kv[1] == "output_var") output_var = kv[2];
              if (kv[1] == "stable") stable = kv[2];
            }
          }
          END {
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s",
                   hidden_naninf, hidden_sat, hidden_var,
                   output_naninf, output_sat, output_var, stable;
          }
        ' "$tmp_log")"
        IFS=$'\t' read -r hidden_naninf hidden_sat hidden_var output_naninf output_sat output_var stable <<< "$guardrails_parsed"

        if [[ -z "$cross_entropy" || -z "$avg_ms" ]]; then
            echo "failed to parse run output for local=$local_mul global=$global_mul mlp=$mlp_mul depth=$depth_power head_weight=$head_weight head_logit=$head_logit" >&2
            cat "$tmp_log" >&2
            exit 1
        fi

        if [[ -z "$hidden_naninf" || -z "$hidden_sat" || -z "$hidden_var" || -z "$output_naninf" || -z "$output_sat" || -z "$output_var" || -z "$stable" ]]; then
            echo "failed to parse guardrails for local=$local_mul global=$global_mul mlp=$mlp_mul depth=$depth_power head_weight=$head_weight head_logit=$head_logit" >&2
            cat "$tmp_log" >&2
            exit 1
        fi

        composite="$(awk -v ce="$cross_entropy" \
                           -v hni="$hidden_naninf" \
                           -v hs="$hidden_sat" \
                           -v hv="$hidden_var" \
                           -v oni="$output_naninf" \
                           -v os="$output_sat" \
                           -v ov="$output_var" \
                           -v st="$stable" 'BEGIN{
            penalty = 0.0;
            if ((hni + oni) > 0) penalty += 1000.0;
            penalty += (hs * 20.0) + (os * 20.0);
            if (hv < 1e-5) penalty += 5.0;
            if (ov < 1e-5) penalty += 5.0;
            if (st != "YES") penalty += 10.0;
            printf "%.6f", ce + penalty;
        }')"

        echo -e "${local_mul}\t${global_mul}\t${mlp_mul}\t${depth_power}\t${head_weight}\t${head_logit}\t${cross_entropy}\t${perplexity:-nan}\t${avg_ms}\t${compile_ms:-nan}\t${load_ms:-nan}\t${hidden_naninf}\t${hidden_sat}\t${hidden_var}\t${output_naninf}\t${output_sat}\t${output_var}\t${stable}\t${composite}\tok" >> "$results_path"
          done
        done
      done
    done
  done
done

echo
echo "Top candidates by composite score:"
sort -t $'\t' -k19,19g -k7,7g -k9,9g "$results_path" | head -n 11 | column -t -s $'\t'
echo
echo "Results written to $results_path"
