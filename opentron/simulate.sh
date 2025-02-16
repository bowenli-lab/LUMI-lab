# shellcheck disable=SC2164

#print pwd
echo "Current directory: $(pwd)"
cd s1_lipid_synthesis
opentrons_simulate -e lipid_synthesis.py
