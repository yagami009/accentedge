#!/bin/bash
# Monitor download progress every 90 minutes
# When all downloads complete, notify user to launch Colab training

LOG="/tmp/downloader.log"
VAANI_LOG="/tmp/vaani_download.log"
CHECK_INTERVAL=5400  # 90 minutes in seconds

echo "Monitor started at $(date)"

while true; do
    echo "=== $(date) ==="

    # Check if downloaders are still running
    MAIN_RUNNING=$(ps aux | grep download_datasets | grep -v grep | wc -l)
    VAANI_RUNNING=$(ps aux | grep download_vaani | grep -v grep | wc -l)

    echo "Main downloader: $([ $MAIN_RUNNING -gt 0 ] && echo 'RUNNING' || echo 'STOPPED')"
    echo "Vaani downloader: $([ $VAANI_RUNNING -gt 0 ] && echo 'RUNNING' || echo 'STOPPED')"

    # Check Phase 3 dataset progress
    HINDI_SIZE=$(du -sh /Volumes/AYUSH_SSD/accentedge_data/phase3/indicvoices_hindi 2>/dev/null | cut -f1 || echo "0")
    TAMIL_SIZE=$(du -sh /Volumes/AYUSH_SSD/accentedge_data/phase3/indicvoices_tamil 2>/dev/null | cut -f1 || echo "0")
    BANGALORE=$(ls /Volumes/AYUSH_SSD/accentedge_data/phase3/vaani_bangalore/ 2>/dev/null | wc -l || echo "0")
    HYDERABAD=$(ls /Volumes/AYUSH_SSD/accentedge_data/phase3/vaani_hyderabad/ 2>/dev/null | wc -l || echo "0")
    CHENNAI=$(ls /Volumes/AYUSH_SSD/accentedge_data/phase3/vaani_chennai/ 2>/dev/null | wc -l || echo "0")
    MUMBAI=$(ls /Volumes/AYUSH_SSD/accentedge_data/phase3/vaani_mumbai/ 2>/dev/null | wc -l || echo "0")
    VIZAG=$(ls /Volumes/AYUSH_SSD/accentedge_data/phase3/vaani_visakhapatnam/ 2>/dev/null | wc -l || echo "0")

    echo "indicvoices_hindi: $HINDI_SIZE"
    echo "indicvoices_tamil: $TAMIL_SIZE"
    echo "vaani_bangalore: $BANGALORE shards"
    echo "vaani_hyderabad: $HYDERABAD shards"
    echo "vaani_chennai: $CHENNAI shards"
    echo "vaani_mumbai: $MUMBAI shards"
    echo "vaani_visakhapatnam: $VIZAG shards"

    # Check if all Vaani districts are complete
    if [ "$BANGALORE" -gt 50 ] && [ "$HYDERABAD" -gt 30 ] && [ "$CHENNAI" -gt 50 ] && [ "$MUMBAI" -gt 50 ] && [ "$VIZAG" -gt 40 ]; then
        echo "=== ALL DOWNLOADS COMPLETE ==="
        echo "Starting Colab training..."

        # Create a Colab session and run training
        colab new --gpu T4 --session accentedge-training 2>&1 | grep -E "READY|Error"

        # Upload and run training script
        colab upload /Users/ayushmh/accentedge/colab_run.py /content/colab_run.py --session accentedge-training 2>&1

        # Execute training
        colab exec --session accentedge-training --file /content/colab_run.py --timeout 14400 2>&1 | tee /tmp/colab_training.log

        echo "Colab training started. Monitor log: /tmp/colab_training.log"
        break
    fi

    # If downloaders are stopped but downloads aren't complete, alert
    if [ "$MAIN_RUNNING" -eq 0 ] && [ "$VAANI_RUNNING" -eq 0 ]; then
        echo "WARNING: Both downloaders stopped but downloads may be incomplete"
        echo "Check logs: $LOG and $VAANI_LOG"
    fi

    # Wait 90 minutes before next check
    echo "Next check in 90 minutes..."
    sleep $CHECK_INTERVAL
done