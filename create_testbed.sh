#!/usr/bin/env bash

echo "Creating testbed module for RN and StateFormers"
mkdir ../rn_testbed

cp translation_tables.yaml ../rn_testbed/translation_tables.yaml
cp ./data/vocab.json ../rn_testbed/vocab.json
cp -r ./bert_modules ../rn_testbed/bert_modules
cp -r ./modules ../rn_testbed/modules
cp -r ./relation_network_modules ../rn_testbed/modules
cp -r ./utils ../rn_testbed/utils
cp ./blender_test/test.py ../rn_testbed/test.py
cp ./blender_test/visualize_test.ipynb ../rn_testbed/visualize_test.ipynb
touch ../rn_testbed/__init__.py

echo "All Ok!"
