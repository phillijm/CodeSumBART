""" meteor.py - Calculate METEOR v1.5 scores.
This implements a Python class as an interface for the official Java
implementation of METEOR v1.5.  For Windows, Mac, & Linux.

You will need to download the official Java implementation of METEOR for this
to work.  Place the download in a folder named "meteor":
        https://www.cs.cmu.edu/~alavie/METEOR/index.html#Download

Author: Jesse Phillips <j.m.phillips@lancaster.ac.uk>
"""
import os


class Meteor():
    def storeData(self,
                  predictions: list,
                  references: list,
                  path: str) -> None:
        """ Saves lists of predictions and references in a format METEOR uses.

        Args:
            predictions (list): the predictions.
            references (list): the references.
            path (string): the filepath you want to save them to.
        """
        with open(f"{path}/pres.txt", 'w', encoding="UTF-8") as fp:
            for x in predictions:
                x = x.replace("\n", "")
                fp.write(f"{x}\n")
        with open(f"{path}/refs.txt", 'w', encoding="UTF-8") as fp:
            for x in references:
                x = x.replace("\n", "")
                fp.write(f"{x}\n")

    def callMeteor(self, path: str) -> None:
        """ Executes the METEOR Java file, saves the outputs of METEOR.

        Args:
            path (string): the filepath you want to save the outputs to.
        """
        import subprocess

        if os.name == "nt":
            pathsep = "\\"
        else:
            pathsep = "/"

        command = "java -Xmx2G -jar"
        meteorLocation = f"meteor{pathsep}meteor-1.5{pathsep}meteor-1.5.jar"
        inputFiles = f".{pathsep}pres.txt .{pathsep}refs.txt"
        args = "-l en -norm"
        commandString = f"{command} {meteorLocation} {inputFiles} {args}"

        _stdout = open(f"{path}/tmpstdout.txt", 'w')
        _stderr = open(f"{path}/tmpstderr.txt", 'w')

        cmd = subprocess.Popen(commandString,
                               stderr=_stderr,
                               stdout=_stdout,
                               cwd=path,
                               shell=True)
        cmd.communicate()

    def getResults(self,
                   path: str,
                   meteorOutput: str,
                   inputLength: int) -> list:
        """ Gets individual METEOR Scores for reference-prediction pairs.

        Args:
            path (string): the filepath you want to save them to.
            meteorOutput (string): file METEOR has saved after callMeteor().
            inputLength (int): the number of reference-prediction pairs.

        Return:
            list: the METEOR scores.
        """
        with open(f"{path}/{meteorOutput}", 'r') as fp:
            lines = [line.rstrip() for line in fp]
            del lines[:11]  # Remove content at start of file
            del lines[inputLength:]  # Remove content after the end
            for cnt in range(len(lines)):
                line = lines[cnt]
                if ':' in line:
                    lines[cnt] = line[line.index(':') + 1:]
                    lines[cnt] = lines[cnt].replace(" ", "").replace("\t", "")
            newLines = []
            for cnt in range(len(lines)):
                newLines.append(float(lines[cnt]))
            return newLines

    def getFinalScore(self, path: str, meteorOutput: str) -> float:
        """ Gets corpus METEOR Score for reference-prediction pairs.

        Args:
            path (string): the filepath you want to save them to.
            meteorOutput (string): file METEOR has saved after callMeteor().

        Return:
            float: the METEOR score.
        """
        with open(f"{path}/{meteorOutput}", 'r') as fp:
            lines = [line.rstrip() for line in fp]
            line = lines[-1]
            if ':' in line:
                line = line[line.index(':') + 1:]
                line = line.replace(" ", "").replace("\t", "")
            return float(line)
