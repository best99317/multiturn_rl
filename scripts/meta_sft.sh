#!/bin/bash

echo "Starting Script 1..."
./scripts/sft_generated_redial.sh # Execute the first script
echo "Script 1 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 2..."
./scripts/sft_generated_redial_copy.sh # Execute the second script
echo "Script 2 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 3..."
./scripts/sft_generated_redial_copy_2.sh # Execute the third script
echo "Script 3 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 4..."
./scripts/sft_generated_redial_copy_3.sh # Execute the fourth script
echo "Script 4 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 5..."
./scripts/dpo_offline_redial_copy_4.sh # Execute the fourth script
echo "Script 5 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 5..."
./scripts/dpo_offline_redial_copy_5.sh # Execute the fourth script
echo "Script 5 finished."

echo "All scripts have finished running."