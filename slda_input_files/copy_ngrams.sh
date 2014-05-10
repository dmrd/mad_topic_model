#!/bin/sh
# Incredible purpose built.
# Copy over files we want to use, clean, and rename
cp $1_etymology_4_author.txt $2_labels
cp $1_etymology_4_model.txt $2_0
cp $1_etymology_4_model.txt $2_0
cp $1_meter_4_model.txt $2_1
cp $1_pos_4_model.txt $2_2
cp $1_syllable_4_model.txt $2_3
cp $1_syllable_count_4_model.txt $2_4
cp $1_word_count_4_model.txt $2_5
python strip_empty.py $2 6
mv $2_0_cleaned $2_0
mv $2_1_cleaned $2_1
mv $2_2_cleaned $2_2
mv $2_3_cleaned $2_3
mv $2_4_cleaned $2_4
mv $2_5_cleaned $2_5
mv $2_labels_cleaned $2_labels
