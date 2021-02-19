Project 1
By:
Mason Brachmann
Ben McCormick
Alberta Yoo
Karim Noorani

Processing tweets to determine the categories, nominees, and winners of the Golden Globes


Things To Install:
In a python enviroment run "pip3 install -r requirements.txt".  The neccessary packages are all listed in requirements.txt

Instructions to Run:
Run python gg_api.py.  This will store the project's generated answer in this file: "(year)answers.json".  It will also store a more-readable file named "(year)_HumanReadableAnswers.txt".
This runs with the assumption that the tweets are stored in a file named: gg(year).json located in the same folder.


How to properly select a year to run:
In gg_api.py's main function, there is a list named year.  Ensure that this list included the years (as ints) that will it will be tested on.
Because there is a change between award names, the award names are hard coded for the years 2013, 2015, 2018, and 2019.  We were unsure of how other year's awards would be named, so testing on different years may lead to a crash.

There are also functions included to show who the best dressed, worst dressed, and funniest people of the golden globes were.  The results are also included in the Human readable text file.