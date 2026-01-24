#!/usr/bin/bash

rm -r datasets/striped_dodgeballs_labeled/train/labels/
rm -r datasets/striped_dodgeballs_labeled/valid/labels/
cp -r datasets/striped_dodgeballs_labeled_copy/train/labels/ datasets/striped_dodgeballs_labeled/train/
cp -r datasets/striped_dodgeballs_labeled_copy/valid/labels/ datasets/striped_dodgeballs_labeled/valid/
