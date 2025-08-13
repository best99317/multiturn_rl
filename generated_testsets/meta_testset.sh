#!/bin/bash

echo "Starting Script 1..."
python generate_testset.py # Execute the first script
echo "Script 1 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 2..."
python generate_testset_copy.py # Execute the second script
echo "Script 2 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 3..."
python generate_testset_copy_2.py # Execute the third script
echo "Script 3 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 4..."
python generate_testset_copy_3.py # Execute the forth script
echo "Script 4 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 5..."
python generate_testset_copy_4.py # Execute the fifth script
echo "Script 5 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 5..."
python generate_testset_copy_5.py # Execute the sixth script
echo "Script 5 finished."

echo "Pausing for 1 minute..."
sleep 1m # Pause for 1 minute

echo "Starting Script 5..."
python generate_testset_copy_6.py # Execute the seventh script
echo "Script 5 finished."

echo "All scripts have finished running."