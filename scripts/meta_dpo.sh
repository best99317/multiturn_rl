#!/bin/bash

echo "Starting Script 1..."
./scripts/dpo_offline_inspired.sh # Execute the first script
echo "Script 1 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 2..."
./scripts/dpo_offline_inspired_copy.sh # Execute the second script
echo "Script 2 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 3..."
./scripts/dpo_offline_inspired_copy_2.sh # Execute the third script
echo "Script 3 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 4..."
./scripts/dpo_offline_inspired_copy_3.sh # Execute the fourth script
echo "Script 4 finished."

echo "All scripts have finished running."