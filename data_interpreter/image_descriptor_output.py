import os

Template_old = """
\\subsection*{{{0} Dataset}}
\\Cref{{fig:{0}:{2}}} shows the results of the {2} feature detector/descriptor on the {0} dataset.
\\begin{{figure}}[H]
	\\centering
	
	\\begin{{subfigure}}[t]{{1\\textwidth}}
		\\includegraphics[width=1\\textwidth]{{{{{1}/img1_2}}.pdf}}
		\\caption{{Correspondences between image 1 and 2}}
	\\end{{subfigure}}
	\\begin{{subfigure}}[t]{{1\\textwidth}}
		\\includegraphics[width=1\\textwidth]{{{{{1}/img1_3}}.pdf}}
		\\caption{{Correspondences between image 1 and 3}}
	\\end{{subfigure}}
	\\begin{{subfigure}}[t]{{1\\textwidth}}
		\\includegraphics[width=1\\textwidth]{{{{{1}/img1_4}}.pdf}}
		\\caption{{Correspondences between image 1 and 4}}
	\\end{{subfigure}}
	\\begin{{subfigure}}[t]{{1\\textwidth}}
		\\includegraphics[width=1\\textwidth]{{{{{1}/img1_5}}.pdf}}
		\\caption{{Correspondences between image 1 and 5}}
	\\end{{subfigure}}
\\begin{{subfigure}}[t]{{1\\textwidth}}
\\includegraphics[width=1\\textwidth]{{{{{1}/img1_6}}.pdf}}
\\caption{{Correspondences between image 1 and 6}}
\\end{{subfigure}}

	
	\\caption{{Results of the {2} feature detector/descriptor on the {0} dataset}}
	\\label{{fig:{0}:{2}}}
\\end{{figure}}


"""

Template = """
\\Cref{{fig:{0}:{2}:1,fig:{0}:{2}:2,fig:{0}:{2}:3,fig:{0}:{2}:4,fig:{0}:{2}:5}} show the results of the {2} feature detector/descriptor on the {0} dataset.
\\begin{{figure}}[H]
    \\includegraphics[width=1\\textwidth]{{{{{1}/img1_2}}.pdf}}
    \\caption{{Correspondences between image 1 and 2}}
    \\label{{fig:{0}:{2}:1}}
\\end{{figure}}
\\begin{{figure}}[H]
    \\includegraphics[width=1\\textwidth]{{{{{1}/img1_3}}.pdf}}
    \\caption{{Correspondences between image 1 and 3}}
    \\label{{fig:{0}:{2}:2}}
\\end{{figure}}
\\begin{{figure}}[H]
    \\includegraphics[width=1\\textwidth]{{{{{1}/img1_4}}.pdf}}
    \\caption{{Correspondences between image 1 and 4}}
    \\label{{fig:{0}:{2}:3}}
\\end{{figure}}
\\begin{{figure}}[H]
    \\includegraphics[width=1\\textwidth]{{{{{1}/img1_5}}.pdf}}
    \\caption{{Correspondences between image 1 and 5}}
    \\label{{fig:{0}:{2}:4}}
\\end{{figure}}
\\begin{{figure}}[H]
    \\includegraphics[width=1\\textwidth]{{{{{1}/img1_6}}.pdf}}
    \\caption{{Correspondences between image 1 and 6}}
    \\label{{fig:{0}:{2}:5}}
\\end{{figure}}

"""

basedir = r"../PartC/out/FeatureDescriptionTesting/"
basedir2 = r"../Code/PartC/out/FeatureDescriptionTesting/"
output = ""
for dataset in os.listdir(basedir):
    if not os.path.isdir(basedir+dataset):
        continue
    for alg in os.listdir(basedir+dataset):
        if alg == "images":
            continue
        if dataset != "leuven":
            continue
        imgdars = basedir2+dataset+"/"+alg
        alg = alg.replace("_", "-")
        outdatasetname = dataset[0].upper() + dataset[1:]
        output += Template.format(outdatasetname, imgdars, alg)

print(output)