#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ./build/train ]]; then
  echo "build/train is missing; run 'make build/train' first." >&2
  exit 1
fi

runs="${ANE_BASELINE_RUNS:-5}"
results_path="${ANE_BASELINE_RESULTS:-build/baseline_v1_runs.tsv}"
mkdir -p "$(dirname "$results_path")"

tmp_log="$(mktemp /tmp/ane-baseline-repeat.XXXXXX)"
trap 'rm -f "$tmp_log"' EXIT

echo -e "run\tcross_entropy\tperplexity\tavg_ms\tthroughput_mb_s\thidden_naninf\thidden_sat\thidden_var\toutput_naninf\toutput_sat\toutput_var\tstable" > "$results_path"

for run in $(seq 1 "$runs"); do
  echo "run $run/$runs" >&2
  ANE_BG_LOG_SURFACE_STATS=1 ./build/train > "$tmp_log" 2>&1

  cross_entropy="$(sed -n 's/.*cross_entropy=\([0-9.]*\).*/\1/p' "$tmp_log" | tail -n 1)"
  perplexity="$(sed -n 's/.*perplexity=\([0-9.]*\).*/\1/p' "$tmp_log" | tail -n 1)"
  avg_ms="$(sed -n 's/.*avg=\([0-9.]*\) ms\/pass.*/\1/p' "$tmp_log" | tail -n 1)"
  throughput="$(sed -n 's/.*throughput=\([0-9.]*\) MB\/s.*/\1/p' "$tmp_log" | tail -n 1)"

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

  if [[ -z "$cross_entropy" || -z "$avg_ms" || -z "$throughput" || -z "$hidden_naninf" || -z "$stable" ]]; then
    echo "failed to parse run $run" >&2
    cat "$tmp_log" >&2
    exit 1
  fi

  echo -e "${run}\t${cross_entropy}\t${perplexity}\t${avg_ms}\t${throughput}\t${hidden_naninf}\t${hidden_sat}\t${hidden_var}\t${output_naninf}\t${output_sat}\t${output_var}\t${stable}" >> "$results_path"
done

echo
echo "Per-run results:"
column -t -s $'\t' "$results_path"
echo
echo "Summary:"
awk -F'\t' '
  NR == 1 { next }
  {
    n++;
    ce[n] = $2 + 0.0;
    ppx[n] = $3 + 0.0;
    ms[n] = $4 + 0.0;
    thr[n] = $5 + 0.0;
    stable_yes += ($12 == "YES");
  }
  END {
    if (n == 0) {
      exit 1;
    }
    mean_ce = mean_ppx = mean_ms = mean_thr = 0.0;
    for (i = 1; i <= n; ++i) {
      mean_ce += ce[i];
      mean_ppx += ppx[i];
      mean_ms += ms[i];
      mean_thr += thr[i];
    }
    mean_ce /= n; mean_ppx /= n; mean_ms /= n; mean_thr /= n;

    var_ce = var_ppx = var_ms = var_thr = 0.0;
    for (i = 1; i <= n; ++i) {
      var_ce += (ce[i] - mean_ce) * (ce[i] - mean_ce);
      var_ppx += (ppx[i] - mean_ppx) * (ppx[i] - mean_ppx);
      var_ms += (ms[i] - mean_ms) * (ms[i] - mean_ms);
      var_thr += (thr[i] - mean_thr) * (thr[i] - mean_thr);
    }
    sd_ce = sqrt(var_ce / n);
    sd_ppx = sqrt(var_ppx / n);
    sd_ms = sqrt(var_ms / n);
    sd_thr = sqrt(var_thr / n);

    printf "runs=%d stable_yes=%d/%d\n", n, stable_yes, n;
    printf "cross_entropy mean=%.4f sd=%.4f\n", mean_ce, sd_ce;
    printf "perplexity    mean=%.2f sd=%.2f\n", mean_ppx, sd_ppx;
    printf "avg_ms        mean=%.3f sd=%.3f\n", mean_ms, sd_ms;
    printf "throughput    mean=%.2f sd=%.2f MB/s\n", mean_thr, sd_thr;
  }
' "$results_path"

echo
echo "Results written to $results_path"
