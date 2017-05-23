import glob
import csv
import os

os.chdir('dataExportForRelease/sqlViews')

if not os.path.exists('R_readable'):
    os.makedirs('R_readable')


for fileName in glob.glob("*.txt"):
    print(fileName + ":\n")

    with open(fileName, 'r') as tsvin:
        tsvout = open('R_readable/R_' + fileName, 'w')
        tsvin = csv.reader(tsvin, delimiter='\t')
        tsvout = csv.writer(tsvout, delimiter='\t')

        for line in tsvin:
            # removes whitespace from sides
            cleanerLine = [entry.strip() for entry in line]
            # removes whitespace within entries
            cleanedLine = [entry.replace(' ', '_') for entry in line]
            # print(cleanedLine)
            tsvout.writerow(cleanedLine)
