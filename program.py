import csv, random, operator, os
import numpy.random as npr

"""
Auteurs : délégués IT BEP 2020-2021 -- Sami ABDUL SATER, Lucas PRIEELS
Contact :
    - samiabdulsater@gmail.com 
    - lucasprieels@gmail.com (à vérifier)

AVANT D'EXECUTER, ASSUREZ VOUS QUE :
    - Les fichiers se nomment "preferences.csv" et "formatSession.csv" et qu'ils ont la même forme que le template proposé.
    - Un dossier "out" est créé dans le même dossier que ce script python.

La sortie du programme est une vingtaine de fichiers nommés "final_xx.csv" qui correspondent aux horaires calculés avec un score pour chaque horaire.
"""
cours = "Rechop"
cours += "/"
test = 0

# A mettre a True quand on voudra continuer ses recherches
load = True



files = ("{}preferences.csv".format(cours), "{}formatSession.csv".format(cours))
if test:
    files = ("preferences2.csv", "formatSession2.csv")

preferenceFile = files[0]
formatFile = files[1]

"""
Réception des dates de la part du Professeur, du fichier formatSession.csv
"""
nPlaces = 0
dateList = []
maxOccupation = dict()
with open(formatFile) as dateFile:
    reader = csv.reader(dateFile)
    rowIndex = 0
    for row in reader:
        if rowIndex != 0:
            date = row[0]
            nMax = int(row[1])
            dateList.append(date)
            maxOccupation[date] = nMax
            nPlaces += nMax
        rowIndex += 1

class Student:
    def __init__(self, matricule, date1, date2, date3):
        self.matricule = matricule
        self.date1 = date1
        self.date2 = date2
        self.date3 = date3
        self.allowedDate = ""
        self.isHappy = False

    def __init__(self, row):
        self.matricule = row[0]
        self.date1 = row[1]
        self.date2 = row[2]
        self.date3 = row[3]
        self.allowedDate = ""
        self.isHappy = False
        self.hasFirst = False
        self.hasSecond = False
        self.hasThird = False

    def copy(self):
        newStud = Student([self.matricule, self.date1, self.date2, self.date3])
        newStud.allowedDate = self.allowedDate
        return newStud

    def __str__(self):
        return self.matricule

    def addDate(self, date):
        self.allowedDate = date

    def appreciation(self):
        if self.allowedDate == self.date1:
            self.isHappy = True
            self.hasFirst = True
            return 100
        elif self.allowedDate == self.date2:
            self.isHappy = True
            self.hasSecond = True
            return 70
        elif self.allowedDate == self.date3:
            self.isHappy = True
            self.hasThird = True
            return 50
        else:
            self.isHappy = False
            return -500

    def whichDate(self):
        if self.allowedDate == self.date1:
            return 1
        elif self.allowedDate == self.date2:
            return 2
        elif self.allowedDate == self.date3:
            return 3
        else:
            return 0


"""
Un chromosome est une liste d'étudiants pour lesquels on a attribué un horaire de passage pour l'exam.
Un chromosome est une "alternative" une proposition de solution au problème.
Donc pour calculer le score de ce chromosome, qui est l'appréciation globale de l'attribution d'horaires, on calcule le score par étudiant (cf. classe student)
"""
class Chromosome:
    def __init__(self, studentList=[], file=""):
        self.occupation = dict()

        if file == "":
            self.studentList = studentList
            for stud in studentList :
                allowedDate = stud.allowedDate
                if (allowedDate not in self.occupation.keys()):
                    self.occupation[allowedDate] = 0
                self.occupation[allowedDate] += 1

            self.score = self.computeScore()
        else:  # constructeur à partir d'un fichier
            self.studentList = []
            with open(file, "r") as csvFile:
                reader = csv.reader(csvFile)
                rowIndex = 0
                for row in reader:
                    if rowIndex != 0:
                        [matricule, allowedDate, date1, date2, date3] = row[:5]
                        stud = Student([matricule, date1, date2, date3])
                        stud.allowedDate = allowedDate
                        if(allowedDate not in self.occupation.keys()) :
                            self.occupation[allowedDate] = 0
                        self.occupation[allowedDate] += 1

                        self.studentList.append(stud)
                    rowIndex += 1

            self.score = self.computeScore()  # Calcule le score et vérifie si les étudiants sont heureux

    def __str__(self):
        matricList = []
        for stud in self.studentList:
            matricList.append(str(stud))
        return str(["{} : {}".format(stud.matricule, stud.allowedDate) for stud in self.studentList])

    def computeScore(self):
        score = 0
        for stud in self.studentList:
            score += stud.appreciation()
        return score

    def copy(self):
        newList =[]
        for student in self.studentList :
            newList.append(student.copy())
        return Chromosome(studentList=newList)

    def save(self, fileName):
        intro = "score : {}/{} (max.), date prévue, date1, date2, date3, date préf?\n".format(self.computeScore(),
                                                                                  len(self.studentList) * 100)
        row = ""
        self.studentList = sorted(self.studentList, key=operator.attrgetter("matricule"))
        for student in self.studentList:
            row += "{},{},{},{},{},{}\n".format(student.matricule, student.allowedDate, student.date1,
                                                     student.date2, student.date3, student.whichDate())

        with open(fileName, "w") as file:
            file.write(intro + row)
            file.close()


    def verifyOccupation(self, maxOccupation):
        """
        Vérifie que le chromosome est un horaire admissible => respecte la capacité par jour
        :param maxOccupation:
        :return: bool
        """
        for stud in self.studentList:
            self.occupation[stud.allowedDate] = 0
        for stud in self.studentList:
            self.occupation[stud.allowedDate] += 1
        for item in self.occupation:
            if self.occupation[item] > maxOccupation[item]:
                return False
        return True

    def equals(self, otherChrom, test=False):
        sortedStudentList = sorted(self.studentList, key=operator.attrgetter("matricule"))
        sortedOtherStudentList = sorted(otherChrom.studentList, key=operator.attrgetter("matricule"))
        for i in range(len(sortedStudentList)):
            if sortedStudentList[i].allowedDate != sortedOtherStudentList[i].allowedDate:
                if (test == True):
                    print("False for student {}".format(sortedOtherStudentList[i].matricule))
                return False
        return True


"""
Réception des préférences via le fichier csv "preferences.csv" où les données commencent à la première ligne
"""
studentList = []

with open(preferenceFile, "r") as file:
    csvReader = csv.reader(file)
    for row in csvReader:
        # colonne = matricule, date1, date2, date3
        studentList.append(Student(row))

"""
Génération d'une population de 10 chromosomes : 10 listes d'étudiants, dans un ordre aléatoire 
"""
population = []
for i in range(10):
    copied = studentList.copy()
    random.shuffle(copied)
    population.append(Chromosome(copied))

"""
Génération d'une population par défaut (utile qu'à la première utilisation de l'algorithme)
Pour chaque chromosome dans la population, on commence par attribuer la date préférée au n°1 de la liste, puis on s'attaque au numéro 2, etc.
S'il y a un conflit (jour déjà complet), on passe à la 2eme date préférée. Etc.

Le problème est qu'ici, l'ordre dans la liste des étudiants compte. Mais ça n'est pas un problème car on construit ici la population initiale.
"""

for chrom in population:
    occupation = dict()  # Dictionnaire qui met à jour l'horaire en comptant combien d'étudiants sont inscrits à chaque date
    for date in dateList:  # Remplissage du dictionnaire
        occupation[date] = 0

    studentList = chrom.studentList.copy()  # Pour ne pas agir sur la liste
    copyStudentsList = []

    for stud in studentList:  # On parcourt la liste dans l'ordre. D'où voir fait 10 listes aléatoires d'étudiants
        copyStud = stud.copy()
        if (occupation[stud.date1] < maxOccupation[stud.date1]):
            copyStud.addDate(stud.date1)
            occupation[stud.date1] += 1
            stud.isHappy = True

        elif (occupation[stud.date2] < maxOccupation[stud.date2]):
            copyStud.addDate(stud.date2)
            occupation[stud.date2] += 1
            stud.isHappy = True

        elif (occupation[stud.date3] < maxOccupation[stud.date3]):
            copyStud.addDate(stud.date3)
            occupation[stud.date3] += 1
            stud.isHappy = True

        else:
            date = random.sample(dateList, 1)[0]
            while occupation[date] >= maxOccupation[date]:
                date = random.sample(dateList, 1)[0]
            copyStud.addDate(date)
            occupation[date] += 1
            stud.isHappy = False

        copyStudentsList.append(copyStud)

    # update population
    newChrom = Chromosome(copyStudentsList)
    population[population.index(chrom)] = newChrom

"""
Two-point crossover
"""

### CROSS OVER (cf. algorithm for Flowshop (INFO-H3000), 000475828, 2020-2021)
"""
Input : ..., studentsToClean : students from the other chromosome, to merge with givenChromosome
"""


def cleanChromosome(givenChromosome, nLeft, nRight, studentsToClean):
    chrom = givenChromosome.studentList.copy()
    studentsToRemove = []

    """
    Here we generate list of students from chromosomes whose ID match the students from matriculesToClean

    Explanation : matriculesToClean comes from chromosome A, and we want to clean chromosome B. Students from chromosome A have a certain assigned date, thus
    they are not the same students as in chromosome B, even with the same ID.
    """
    for stud in studentsToClean:
        for elem in givenChromosome.studentList:
            if elem.matricule == stud.matricule:
                studentsToRemove.append(elem)

    for stud in studentsToRemove:
        chrom.pop(chrom.index(stud))

    """
    Now we split our chromosome in two parts : right and left zone of the crossover.
    We fill the right zone by pushing all present genes to the left, then filling the right part (still in the zone
    located at the right of the exchange zone !) with the first gene of the "chromosome of lefts", until we reach
    length of the chromosome
    Then, we fill the left part with the remaining
    """
    listRight = []
    sizeListRight = len(givenChromosome.studentList) - nRight
    listLeft = []
    sizeListLeft = nLeft

    # Extracting genes from remaining chromosome to form the right part
    for gene in reversed(chrom):
        if gene in givenChromosome.studentList[nRight:]:
            listRight.append(gene)
            chrom.pop(-1)
    listRight.reverse()

    # Extracting from remaining chromosome to build the remaining right part with the first genes of the remaining chrom
    while len(listRight) < sizeListRight:
        listRight.append(chrom[0])
        chrom.pop(0)

    # Remaining genes are the left part of the exchange
    listLeft = chrom

    # Generating list of students from studentsToClean
    studentsToMerge = []
    for stud in studentsToClean:
        studentsToMerge.append(stud)

    return Chromosome(listLeft + studentsToMerge + listRight)


def crossover(chrom1, chrom2):
    size = len(chrom1.studentList)
    x, y = choseRandom(size)
    toCleanFrom1 = [stud.copy() for stud in chrom2.studentList.copy()[x:y]]
    toCleanFrom2 = [stud.copy() for stud in chrom1.studentList.copy()[x:y]]

    firstChild = cleanChromosome(chrom1, x, y, toCleanFrom1)
    secondChild = cleanChromosome(chrom2, x, y, toCleanFrom2)

    # Si les enfants sont non-admissibles, on recommence le cross-over jusqu'à ce qu'ils le soient
    # et si au bout de 1000 tentatives, aucun accord trouvé, alors on dit qu'on n'a pas pu faire le croisement
    nAttempts = 0
    while firstChild.verifyOccupation(maxOccupation) == False or secondChild.verifyOccupation(maxOccupation) == False:
        if nAttempts > 1000:
            return False
        x, y = choseRandom(size)
        toCleanFrom1 = [stud.copy() for stud in chrom2.studentList.copy()[x:y]]
        toCleanFrom2 = [stud.copy() for stud in chrom1.studentList.copy()[x:y]]

        firstChild = cleanChromosome(chrom1, x, y, toCleanFrom1)
        secondChild = cleanChromosome(chrom2, x, y, toCleanFrom2)
        nAttempts += 1
    return [firstChild, secondChild]


def choseRandom(size):
    [x, y] = random.sample(range(size), 2)
    # Putting order if x > y
    if x > y:
        t = x
        x = y
        y = t
    return x, y


"""
Les chromosomes avec une faible appréciation sont mutés : échange de date entre un étudiant et un autre
"""


def mutation(chrom):

    size = len(chrom.studentList)
    (a, b) = random.sample(range(size), 2)

    # Trouver les deux étudiants
    studentA = chrom.studentList[a]
    studentB = chrom.studentList[b]

    # Echange des dates
    tempDate = studentA.allowedDate
    studentA.allowedDate = studentB.allowedDate
    studentB.allowedDate = tempDate

    # Met à jour le score 
    chrom.score = chrom.computeScore()
    return chrom


def mutatePopulation(population, nToMutate):
    sortedPop = sorted(population, key=operator.attrgetter("score"))

    for i in range(int(nToMutate)):
        chrom = sortedPop[i].copy()

        sortedPop[i] = mutation(chrom)

    return sortedPop


"""
Processus de sélection
"""


def selectOne(population):
    max = sum(chromosome.score for chromosome in population)
    selection_probs = [(chrom.score) / max for chrom in population]
    return population[npr.choice(len(population), p=selection_probs)]


def selection(population, nSelect):
    newPop = []
    nAdded = 0

    selectionTable = dict()
    for chrom in population:
        selectionTable[chrom] = False
    while nAdded < nSelect:
        selectedChrom = selectOne(population)

        # on change de chromosome sélectionné si il a déjà été pris
        while selectionTable[selectedChrom] == True:
            selectedChrom = selectOne(population)

        selectionTable[selectedChrom] = True
        newPop.append(selectedChrom)

        nAdded += 1
    return newPop


# Other version of selection
def otherSelection(population, nSelect):
    # il suffit de mettre reverse=False pour faire un tri dans le sens inverse et sélectionner les pires candidats
    sortedPop = sorted(population, reverse=True, key=operator.attrgetter("score"))
    selected = []
    i = 0
    nSelected = 0
    while (nSelected < nSelect and i < len(sortedPop)):
        # on doit vérifier si le chromosome ne figure pas déjà dans ceux sélectionnés (même si dans un autre ordre)
        # ça revient à vérifier si la population ne contient pas deux chromosomes équivalents
        hasEquivalent = False
        for chrom in selected:
            if (sortedPop[i].equals(chrom)):
                hasEquivalent = True

        if (not hasEquivalent):
            selected.append(sortedPop[i])
            nSelected += 1
        i += 1
    return selected


# Le cross over, la mutation, et la sélection sont faits.


def makeNewGenFromPopulation(population, nToSelect, genSize, percentToMutate):
    """ Construit une nouvelle génération à partir d'une population donnée. Effectue la sélection, le cross-over, et la mutation sur les pires candidats.

    Args:
        population ([chromosome]): une liste de chromosomes (cf. classe chromosome)
        nToSelect (integer) : nombre de chromosomes à sélectionner de chaque population pour former la génération suivante de taille genSize
        genSize (integer): taille de la future génération
        percentToMutate (integer) : pourcentage de la population à faire muter
    """

    nChildren = 0

    """
     Etape 1 : selection des meilleurs
    """
    selected = otherSelection(population, nToSelect)

    """
     Etape 2 : cross-over pour obtenir genSize enfants.
        - prend 2 chromosomes au hasard, les fait se reproduire
        - prend un des deux enfants au hasard
    """
    newPop = []
    while (nChildren < genSize):
        popSize = len(selected)
        (a, b) = random.sample(range(popSize), 2)

        chromosomeA = selected[a]
        chromosomeB = selected[b]

        whichChild = random.randint(1, 2)

        children = crossover(chromosomeA, chromosomeB)

        # Si le cross over a réussi (cf. le "return False" dans le cross over) alors on l'ajoute, sinon on réessaye
        if children != False  :

            child = children[whichChild - 1]
            newPop.append(child)
            nChildren += 1

    """
     Etape 3 : la mutation sur les candidats avec la plus faible appréciation

    """
    nToMutate = percentToMutate * popSize
    newPop = mutatePopulation(newPop, nToMutate)
    return newPop


def routine(initialPopulation, nIterations, nToSelect, sizeOfGens, mutation, diffToConverge):
    """Considère une population de départ, et construit des générations futures jusqu'à convergence ou arrêt de l'algorithme.
    Construit aussi le dossier "out" qui sera remplis d'horaires.
    Il met dans un même dossier tous les horaires qui ont le même score.

    L'algorithme garde aussi en mémoire l'horaire avec le plus grand score.

    Args:
        initialPopulation ([chromomsome]): une liste d'objets "chromosome" qui constitue la population initiale
        nIterations (int): le nombre de générations maximal à générer   
        nToSelect (int): le nombre de chromosomes à sélectionner de chaque génération pour construire la suivante   
        sizeOfGens (int): la taille de la génération à générer à chaque étape
        mutation (float): pourcentage de "faibles" de la population qui seront concernés par la mutation
        diffToConverge (int): un nombre qui permettra d'évaluer si l'algorithme a convergé ou stagne autour d'un extremum global. Si au bout de diffToConverge générations, les chromosomes ont la même évaluation, c'est qu'on a convergé.
    """
    n = 0
    maxScore = len(initialPopulation[0].studentList) * 100
    nSameEval = 0
    nDiffEval = 0
    convergence = False
    oldEval = 0

    sameScore = dict()

    while (n < nIterations and convergence == False):

        newPop = makeNewGenFromPopulation(initialPopulation, nToSelect, sizeOfGens, mutation)

        topChromosome = sorted(newPop, key=operator.attrgetter("score"), reverse=True)[0]
        topChromosome.score = topChromosome.computeScore()
        # TODO : reconnaitre un maximum et le sauvegarder
        newEval = topChromosome.score
        print("Itération {} : score {}/{}".format(n, newEval, maxScore))
        if (newEval >= maxScore * 1):  # Ex : 19500/20000
            convergence = True

        if newEval not in sameScore.keys() :
            sameScore[newEval] = 0
        if(sameScore[newEval] == 0) :
            topChromosome.save("{}out/score{}.csv".format(cours,newEval))
        sameScore[newEval] += 1
        """
        Analyzing convergence. Increase mutation in case of convergence.
        Decreases mutation in case of different generations for 5 generations straight.
        """
        if (oldEval == newEval):
            nDiffEval = 0
            nSameEval += 1
            if (nSameEval > diffToConverge):
                print("Increasing mutation from {} to {}".format(mutation, mutation + 0.2))
                mutation += 0.2
                if (mutation >= 1):
                    mutation = 0.8

        else:
            nSameEval = 0
            nDiffEval += 1
            if (nDiffEval > 2):
                mutation = 0.2  # Reset mutation increase
            oldEval = newEval
        initialPopulation = newPop
        n += 1
    topChromosomes = otherSelection(newPop, 20)
    return topChromosomes

if load :
    oldChroms = []

    dir = "{}out".format(cours)
    for file in sorted(os.listdir(dir), reverse=True) :
        if not os.path.isdir("{}/{}".format(dir,file)) :
            oldChroms.append(Chromosome(file="{}/{}".format(dir,file)))
    population = oldChroms

newPop = makeNewGenFromPopulation(population, len(population), 100, 0.2)

# Paramètres
nIterations = 100
nToSelect = 10
sizeOfGens = 500
mutationFactor = 0.2
diffToConverge = 5





def run_algo():
    # La ligne qui fait tout tourner
    topChroms = routine(newPop, nIterations, nToSelect, sizeOfGens, mutationFactor, diffToConverge)

    sortedChroms = sorted(topChroms, key=operator.attrgetter("score"), reverse=True)
    dir = "{}out/final".format(cours)
    if not os.path.isdir(dir) :
        os.mkdir(dir)

    for chrom in sortedChroms :
        chrom.score = chrom.computeScore()
        chrom.save("{}out/final/score{}.csv".format(cours, chrom.score))

run_algo()
run_algo()
run_algo()
run_algo()
run_algo()