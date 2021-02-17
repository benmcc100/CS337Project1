Project 1
By:
Mason Brachmann
Ben McCormick
Alberta Yoo
Karim Noorani

Processing tweets to determine the categories, nominees, and winners of the Golden Globes


Things To Install:
In a python enviroment run "python install -r requirements.txt".  The neccessary packages are all listed in requirements.txt

Run python gg_api.py.  This will store the project's generated answer in this file: "answers.json".  It will also store a more-readable file named "HumanReadableAnswers.json".
This runs with the assumption that the tweets are stored in a file named: gg(year).json located in the same folder.
In gg_api.py's main function, there is a list named year.  Ensure that this list included the years that will it will be tested on.

Code inside the function getSolutionsFromPositions was taken from https://stackoverflow.com/questions/63450423/how-to-find-proper-noun-using-spacy-nlp by user T. Jeanneau
This code will find take the doc and return a list of pos search that concates indices that are concurrent
