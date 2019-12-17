import os

Template = """
\\subsection*{{{0} Dataset}}
\\Cref{{fig:{0}:{2}}} shows the results of the {2} feature detector on the {0} dataset.
\\begin{{figure}}[H]
	\\centering
	
	\\begin{{subfigure}}[t]{{0.45\\textwidth}}
		\\includegraphics[width=1\\textwidth]{{{{{1}/img1}}.pdf}}
		\\caption{{Image 1}}
	\\end{{subfigure}}
	\\begin{{subfigure}}[t]{{0.45\\textwidth}}
		\\includegraphics[width=1\\textwidth]{{{{{1}/img2}}.pdf}}
		\\caption{{Image 2}}
	\\end{{subfigure}}
	\\begin{{subfigure}}[t]{{0.45\\textwidth}}
		\\includegraphics[width=1\\textwidth]{{{{{1}/img3}}.pdf}}
		\\caption{{Image 3}}
	\\end{{subfigure}}
	\\begin{{subfigure}}[t]{{0.45\\textwidth}}
		\\includegraphics[width=1\\textwidth]{{{{{1}/img4}}.pdf}}
		\\caption{{Image 4}}
	\\end{{subfigure}}
\\begin{{subfigure}}[t]{{0.45\\textwidth}}
\\includegraphics[width=1\\textwidth]{{{{{1}/img5}}.pdf}}
\\caption{{Image 5}}
\\end{{subfigure}}
\\begin{{subfigure}}[t]{{0.45\\textwidth}}
\\includegraphics[width=1\\textwidth]{{{{{1}/img6}}.pdf}}
\\caption{{Image 6}}
\\end{{subfigure}}
	
	\\caption{{Results of the {2} feature detector on the {0} dataset}}
	\\label{{fig:{0}:{2}}}
\\end{{figure}}


"""

basedir = r"../PartC/out/FeatureDetectionTesting/"
basedir2 = r"../Code/PartC/out/FeatureDetectionTesting/"
output = ""
for dataset in os.listdir(basedir):
    if not os.path.isdir(basedir+dataset):
        continue
    for alg in os.listdir(basedir+dataset):
        imgdars = basedir2+dataset+"/"+alg
        outdatasetname = dataset[0].upper() + dataset[1:]
        output += Template.format(outdatasetname, imgdars, alg)

print(output)