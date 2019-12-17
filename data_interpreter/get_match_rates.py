import json
import numpy as np


datafile = r"..\PartC\out\FeatureDetectionTesting\match_rates.json"

with open(datafile, "r") as fp:
    data = json.load(fp)


for datasetname in data:
    outdatasetname = datasetname[0].upper()+datasetname[1:]
    print(outdatasetname)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for feature_detector_name in sorted(data[datasetname],
                                        key=lambda k: np.average(data[datasetname][k]["MatchRates"]),
                                        reverse=True):
        linestring = r"\textbf{"+feature_detector_name+"}"
        for matchrate in data[datasetname][feature_detector_name]["MatchRates"]:
            linestring += "&\t" + str(round(matchrate*100, 2))
        print(linestring+"&\t" + str(round(np.average(data[datasetname][feature_detector_name]["MatchRates"])*100, 2))+r"\\\hline")
