import json
import numpy as np


datafile = r"..\PartC\out\FeatureDescriptionTesting\match_rates.json"

with open(datafile, "r") as fp:
    data = json.load(fp)


for datasetname in data:
    outdatasetname = datasetname[0].upper()+datasetname[1:]
    print(outdatasetname)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for feature_detector_name in sorted(data[datasetname],
                                        key=lambda k: np.nanmean(np.array(data[datasetname][k]['correct_match_count']) /
                                                                 data[datasetname][k]['match_count']),
                                        reverse=True):
        linestring = r"\textbf{"+feature_detector_name+"}"
        vals = 0.0
        valcnt = 0.0
        for mc, cmc in zip(data[datasetname][feature_detector_name]['match_count'],
                           data[datasetname][feature_detector_name]['correct_match_count']):
            if mc>0:
                vals += 100*cmc / mc
            valcnt += 1
            linestring += "&\t" + ("0" if mc == 0 else str(round(cmc/mc*100, 2)))
        print(linestring+"&\t" + str(round(vals/valcnt, 2))+r"\\\hline")
"""
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FP")
    for feature_detector_name in sorted(data[datasetname],
                                        key=lambda k: np.average(
                                            (data[datasetname][k]['match_count'] -
                                             np.array(data[datasetname][k]['correct_match_count'])) /
                                            data[datasetname][k]['match_count'])
                                        ):
        linestring = r"\textbf{"+feature_detector_name+"}"
        vals = 0.0
        valcnt = 0.0
        for mc, cmc in zip(data[datasetname][feature_detector_name]['match_count'],
                           data[datasetname][feature_detector_name]['correct_match_count']):
            if mc>0:
                vals += 100*(mc-cmc)/mc
            valcnt += 1
            linestring += "&\t" + ("0" if mc == 0 else str(round((mc-cmc)/mc*100, 2)))
        print(linestring+"&\t" + str(round(vals/valcnt, 2))+r"\\\hline")
"""