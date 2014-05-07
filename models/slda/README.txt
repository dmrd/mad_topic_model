Hello Gentlemen. We did. Here is how to run the SLDA CODE.

Here is the usage:

option 1: slda [est] [word_types] [data] [label] [settings] [alpha] [random/seeded/model_path] [directory] [topic#1] [topic #2] ...

option 2:   slda [inf] [word_types]  [data] [label] [settings] [model] [directory];
 
[est] indicates that we will be doing parameter estimation 
[inf] indicates that we are inferening topics after doing parameter estimation 

[word_types] is the number of different n-grams types we are using

[data] is the base name of the data files. if "file.txt" is the input, 
then then the data files will be labeled "file.txt_0", "file.txt_1",
..."file.txt_t", where t = [word_types]. NOTE THAT ALL DOCUMENTS MUST
BE LAID OUT IN THE SAME ORDER IN EACH FILE!!!!

[label] is the name of the file containing document labels. labels must start at 0 and go to the number of classes-1

[settings] is the location of the settings file

[alpha] is the initial value of the prior before MAP fitting

[random/seeded/model_path] is the method used for initializing parameters

[directory] is the location of the written files

[topic#1], [topic#2], dots... are the number of topics for each word type

As of now, only [est] and [random] options are supported. 