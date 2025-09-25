#!/bin/bash

# This script runs the lm.py command on two different input files.

echo "nom_vtran"
time python 1word1mask.py deasy/vtrans.txt nom_vtran -o deasy/nom_vtran.tsv

echo "nom_vintran"
time python 1word1mask.py deasy/vintrans.txt nom_vintran -o deasy/nom_vintran.tsv

echo "agent_vintran"
time python 1word1mask.py deasy/vintrans.txt agent_vintran -o deasy/agent_vintran.tsv

echo "agent_vtran"                                                                                                                                        
time python 1word1mask.py deasy/vtrans.txt agent_vtran -o deasy/agent_vtran.tsv

echo "evt_vintran"
time python 1word1mask.py deasy/vintrans.txt evt_vintran -o deasy/evt_vintran.tsv

echo "evt_vtran"
time python 1word1mask.py deasy/vtrans.txt evt_vtran -o deasy/evt_vtran.tsv

echo "participleAdj_vintran"
time python 1word1mask.py deasy/vintrans.txt participleAdj_vintran -o deasy/participleAdj_vintran.tsv

echo "participleAdj_vtran"
time python 1word1mask.py deasy/vtrans.txt participleAdj_vtran -o deasy/participleAdj_vtran.tsv

echo "causative_vintran"
time python 1word1mask.py deasy/vintrans.txt causative_vintran -o deasy/causative_vintran.tsv

echo "causative_vtran"
time python 1word1mask.py deasy/vtrans.txt causative_vtran -o deasy/causative_vtran.tsv

echo "END"
