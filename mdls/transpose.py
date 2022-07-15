import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from path import Path
import pickle
import math
import scipy.stats as st
import sys
from typing import Tuple
from sklearn import impute  # .IterativeImputer
from sklearn import preprocessing
from sklearn import decomposition

TOP_LEVEL_OF_FILTERING = 80  # in percent
NAME_FILE_SCORE_UNIVARIATE_ANALYSIS = 'Score_for_univariate_analysis.csv'
NAME_FILE_GRADE_UNIVARIATE_ANALYSIS = 'Grade_for_univariate_analysis.csv'
NAME_FILE_BOTH_UNIVARIATE_ANALYSIS = 'Both_for_univariate_analysis.csv'
NOM_FICHIER_CSV = 'fr.openfoodfacts.org.products.csv'
NOM_FICHIER_FILTRE_CSV = 'openfoodfact.reduced.csv'
NOM_FICHIER_PICKLE = 'fr.openfoodfacts.org.products.dat'
SEUILS_DE_VALEURS: dict = {
    'energy_100g': 3900,
    'fat_100g': 100,
    'saturated-fat_100g': 92.6,
    'trans-fat_100g': 37,
    'cholesterol_100g': 3.1,
    'carbohydrates_100g': 99.8,
    'sugars_100g': 99.6,
    'fiber_100g': 43.5,
    'proteins_100g': 39,
    'salt_100g': 4.9,
    'sodium_100g': 4.9,
    'vitamin-a_100g': 0.2,
    'vitamin-c_100g': 3.2,
    'calcium_100g': 2,
    'iron_100g': 0.087
}
MAX_VALUES_LIST = []
for key, value in SEUILS_DE_VALEURS.items():
    MAX_VALUES_LIST.append(value)


def replace_null_values(data):
    imp_mean = impute.IterativeImputer(random_state=42, min_value=0., max_value=MAX_VALUES_LIST)
    imp_mean.fit(data)
    X_imp = imp_mean.transform(data)
    index = data.index
    columns_dict: dict = {}
    for i, col in enumerate(data.columns):
        columns_dict.update({i: col})
    Xy_df = pd.DataFrame(X_imp, index=index)
    Xy_df = Xy_df.rename(columns=columns_dict)
    return Xy_df


def repair_db():
    """
    La base de données contenant des lignes mal-encodées, nous devons la charger avec plus de mémoire pour
    ensuite sélectionner les colonnes mal-typé apparue dans le terminal sous le nom de Dtypes warning.
    Enfin nous rechargeons la base de données en ne prenant pas la colonne 'code' comme index volontairement,
     et en passant le type des colonnes douteuse en argument puis nous enregistrons la base en excluant les valeurs
    indexes nulles car elles sont des lignes dupliquées.
    :return: void
    """
    data_broken = pd.read_csv('./' + NOM_FICHIER_CSV, sep='\t', low_memory=False)
    bad_list: list = [0, 3, 5, 19, 20, 24, 25, 26, 27, 28, 35, 36, 37, 38, 39, 48]
    bad_columns_list: list = [list(data_broken.columns)[i] for i in bad_list]
    bad_columns_dict: dict = dict()
    for col in bad_columns_list:
        bad_columns_dict.update({col: 'str'})
    path_name = Path('./' + NOM_FICHIER_CSV)
    data = pd.read_csv(path_name, sep='\t', low_memory=False,
                       dtype=bad_columns_dict, engine='c')

    path_name_pickle = Path('./' + NOM_FICHIER_PICKLE)
    write_to_pickle(path_name_pickle, data)


def write_to_pickle(path, item):
    """
    Fonction pour écrire un objet en base sérialisée avec pickle,
    sous la forme d'un fichier au format dat.
    :param path: un chemin de destination
    :param item: un fichier source
    :return: void
    """
    with open(path, 'wb') as file:
        pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)


def drop_dt_col(data: pd.DataFrame) -> pd.DataFrame:
    """
    Élimination des colonnes datetimes et timestamp. La base de données utilise les suffixes
    _t et _datetime pour signaler une colonne de temps. Cette fonction en tire partie.
    :param data: notre base de données.
    :return: data: la base sans les colonnes datetime et timestamp.
    """
    data_returned = data.copy()
    for col in list(data_returned.columns):
        if col.endswith('_t') | col.endswith('_datetime'):
            data_returned.drop(col, axis=1, inplace=True)
    return data_returned


def pourcent_of_null(data: pd.Series) -> float:
    """
    Renvoie le pourcentage de valeurs nulles dans une Series de pandas.
    :param data: Series de pandas
    :return: float: le pourcentage
    """
    if data.ndim == 1:
        return round((~data.notnull()).sum() * 100 / len(data), 2)
    else:
        print('le paramètre est-il vraiment une Series de pandas ?')


def pourcent_of_duplicated(data: pd.Series) -> float:
    """
    Renvoie le pourcentage de valeurs dupliquées dans une Series de pandas.
    :param data: Series de pandas
    :return: float: le pourcentage
    """
    if data.ndim == 1:
        return round((~data.duplicated()).sum() * 100 / len(data), 2)
    else:
        print('le paramètre est-il vraiment une Series de pandas ?')


def iqr(data: pd.Series) -> float:
    """
    Renvoie le pourcentage de valeurs au-dehors de l'intervalle de confiance externe.
    :param data: une Series de pandas.
    :return: float: le pourcentage.
    """
    if data.ndim == 1:
        is_a_series_of_str = True in map((lambda x: type(x) == str), data)
        if not is_a_series_of_str:
            iqr_value = round((data.quantile(0.75) - data.quantile(0.25)) * 3, 2)
            return iqr_value


def pourcent_outside_3iqr(data: pd.Series) -> float:
    """
    Renvoie le pourcentage de valeurs au-dehors de l'intervalle de confiance externe.
    :param data: une Series de pandas.
    :return: float: le pourcentage.
    """
    if data.ndim == 1:
        is_a_series_of_str = True in map((lambda x: type(x) == str), data)
        if not is_a_series_of_str:
            iqr = round((data.quantile(0.75) - data.quantile(0.25)) * 3, 2)
            if len(data) > 0:
                pct_outside_confidence_interval = round(len(data.loc[(data > iqr) | (data < 0)]) / len(data) * 100, 2)
            else:
                pct_outside_confidence_interval = 0
            return pct_outside_confidence_interval


def type_of_series(data: pd.Series) -> str:
    """
    Type d'une Series de pandas parmi les trois possibilités suivante :
    -> une string
    -> un float
    -> un timestamp
    Mais ne retourne que les type string ou float pour plus de praticité avec l'instanciateur des
    Series de pandas : voir la fonction select_iqr_interval().
    :param data:
    :return:
    """
    is_a_series_of_str = True in map((lambda x: type(x) == str), data)
    is_a_series_of_float = True in map((lambda x: type(x) == float), data)
    is_a_series_of_timestamp = True in map((lambda x: type(x) == pd.Timestamp), data)
    if is_a_series_of_float:
        return 'float'
    elif is_a_series_of_str:
        return 'str'
    elif is_a_series_of_timestamp:
        return 'str'
    else:
        return 'object'


def select_iqr_interval(data: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un cadre de données vide comprenant des Series de pandas soit de dtype string, soit de dtype float.
    Ensuite, dans une boucle, seulement pour les colonnes en _n ou en _100g, mais qui ne contiennnent pas le
    mot 'score' de nutri-score (car cette colonne doit rester, intégralement), l'iqr est calculé.
    Le tout est déposé dans le nouveau cadre de donnée avec le nutri-score.
    :param data: la base de données.
    :return: data: la base de données avec les colonnes de valeurs numéraires.
    """
    col_list: list = range(data.shape[1])  # number of columns
    columns_list: list = [list(data.columns)[i] for i in col_list]
    columns_dict: dict = dict()
    is_of_type: str = ''
    for col in columns_list:  # Loop to instanciate the returned frame
        is_of_type = type_of_series(data[col])
        is_a_numeric_col = col.endswith('_n') | col.endswith('_100g')
        if is_a_numeric_col:
            columns_dict.update({col: pd.Series([], dtype=is_of_type)})
    data_returned = pd.DataFrame(index=list(data.index), columns=columns_dict)
    for col in list(data.columns):
        is_a_numeric_col = col.endswith('_n') | col.endswith('_100g')
        if is_a_numeric_col & (('score-fr' not in col) & ('score-uk' not in col)):
            try:
                q3: float = float(data[col].quantile(0.75, 'lower'))
                q1: float = float(data[col].quantile(0.25, 'higher'))
                iqr_x_3 = round((q3 - q1) * 3, 2)
                data_returned[col] = data[col].loc[(data[col] < iqr_x_3) & (data[col] >= 0)]
            except ValueError as error:
                raise ValueError(error)
        elif is_a_numeric_col & ('score-fr' in col):
            data_returned[col] = data[col]
        elif not is_a_numeric_col & ('grade_fr' in col):
            data_returned[col] = data[col]
        else:
            continue
    return data_returned


def select_low_null_value_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cette fonction prend la dataframe et ne garde que les colonnes qui n'ont pas plus de n pourcents (Top level
    of filtering) et renvoie une dataframe, qui, en plus ne contient pas de nutri-score nul"""
    columns: list = list(data.columns)
    filtered_columns: list = []
    for col in columns:
        is_a_numeric_col = col.endswith('_n') | col.endswith('_100g')
        if is_a_numeric_col & (('score-fr' not in col) & ('score-uk' not in col)):
            if pourcent_of_null(data[col]) < TOP_LEVEL_OF_FILTERING:
                filtered_columns.append(col)
        elif ('score-fr' in col) | ('grade_fr' in col):
            filtered_columns.append(col)

    filtered_columns.append('code')

    data_returned = data[filtered_columns].copy()
    data_returned['nutrition_grade_fr_n'] = data_returned['nutrition_grade_fr'].map({'a': 1, 'b': 2, 'c': 3,
                                                                                     'd': 4, 'e': 5})
    data_returned['nutrition_grade_fr_n'] = data_returned['nutrition_grade_fr_n'].convert_dtypes(convert_integer=True)

    data_returned = data_returned[~data_returned['code'].duplicated()]
    data_returned = data_returned[~data_returned['code'].isnull()]
    data_returned[~data_returned['nutrition-score-fr_100g'].isnull()].to_csv(NOM_FICHIER_FILTRE_CSV, sep='\t')

    return data_returned[~data_returned['nutrition-score-fr_100g'].isnull()]


def abs_and_bound_values(data):
    """
    Rectification des valeurs d’une data frame, ne prenant que les valeurs non-nulles.
    :param data: Cadre de données
    :return: data: Cadre de données
    """
    rec: dict = SEUILS_DE_VALEURS
    for col in data.columns:
        if 'score' not in col:
            data[col][~data[col].isnull()] = data[col][~data[col].isnull()].apply(lambda x: abs(x) if x < 0 else x)
            data[col][~data[col].isnull()] = data[col][~data[col].isnull()].apply(
                lambda x: x if (x <= rec[col]) else rec[col])
    return data


def thresholding(data):
    """
    Renvoie un tableau pour présenter les seuils choisis (Lim), le maximum et le minimum de la colonne, et le nombre
    de valeurs au-dessus du seuil, puis le nombre de valeurs négatives.
    :param data: pandas DataFrame
    :return: void
    """
    rec: dict = SEUILS_DE_VALEURS
    dataframe = pd.DataFrame({'columns': pd.Series([], dtype=str),
                              'non-nulls': pd.Series([], dtype=int),
                              'limits': pd.Series([], dtype=float),
                              'max': pd.Series([], dtype=float),
                              'min': pd.Series([], dtype=float),
                              'outliers': pd.Series([], dtype=int),
                              'negatives': pd.Series([], dtype=int),
                              'kurtosis': pd.Series([], dtype=float),
                              'skewness': pd.Series([], dtype=float),
                              'mode': pd.Series([], dtype=float),
                              'median': pd.Series([], dtype=float)
                              })
    format_str = '{:^3}{:<1}{:^42}{:<1}{:^7}{:<1}{:^5}{:<1}{:^5}{:<1}{:^5}{:<1}{:^5}{:<1}{:^5}{:<1}{:^5}{:<1}{:^5}{:<1}{:^5}{:<1}{:^5}{:<1}'
    # Header
    # print(format_str
    #       .format('#', '|', 'Column', '|', 'NotNl', '|', 'Lim', '|', 'Max', '|', 'Min', '|', 'Out', '|', 'Neg', '|',
    #               'Kurt', '|', 'Skew', '|', 'Mode', '|', 'Medn', '|'))
    # print(
    #     format_str.format('---', '|', '-----', '|', '-----', '|', '---', '|', '---', '|', '---', '|', '---', '|', '---',
    #                       '|', '---', '|', '---', '|', '---', '|', '---', '|'))

    # dtypes information
    dtypes_uniques = set()  # Collection of unique elements
    dtypes_listes = []  # list of complete data columns
    memory_usage = 0  # in MB
    nb_values_to_modify: int = 0
    nb_values_not_null: int = 0
    nb_values_is_null: int = 0
    for it, col in enumerate(data.columns):  # Feed the set and the list
        is_a_str_series = True in map((lambda val: type(val) == str), data[col])
        max_col = str(data[col].max()) if not is_a_str_series else 'None'
        min_col = str(data[col].min()) if not is_a_str_series else 'None'
        try:
            out_col = data[col][data[col] > rec[col]].count()
        except KeyError as error:
            out_col = 0
        print(col)
        neg_col = data[col][data[col] < 0].count()
        nb_values_to_modify += out_col + neg_col
        nb_values_not_null += data[col].count()
        try:
            nb_values_is_null += data[col].isnull().value_counts()[True]
        except KeyError as error:
            nb_values_is_null += 0
        # print(str(data[col].max()))
        dtypes_listes.append(str(data[col].dtype))
        if str(data[col].dtype) not in dtypes_uniques:
            dtypes_uniques.add(str(data[col].dtype))
        # Table body
        # print(
        #     '{:^3}{:<1}{:^42}{:<1}{:^7.7}{:<1}{:^5.5}{:<1}{:^5.5}{:<1}{:^5.5}{:<1}{:^5.5}{:<1}{:^5.5}{:<1}{:^5.5}{:<1}{:^5.5}{:<1}{:^5.5}{:<1}{:^5.5}{:<1}'.format(
        #         str(it), '|', col, '|',
        #         str(data[col].count()), '|',
        #         str(rec.get(col)), '|',
        #         max_col, '|',
        #         min_col, '|',
        #         str(out_col), '|',
        #         str(neg_col), '|',
        #         str(data[col].kurt()), '|',
        #         str(data[col].skew()), '|',
        #         str(data[col].mode()), '|',
        #         str(data[col].median()), '|'
        #     ))
        dataframe = dataframe.append({'columns': col, 'non-nulls': data[col].count(), 'limits': rec.get(col),
                                      'max': max_col, 'min': min_col, 'outliers': out_col,
                                      'negatives': neg_col,
                                      'kurtosis': data[col].kurt(),
                                      'skewness': data[col].skew(),
                                      'mode': data[col].mode()[0],
                                      'median': data[col].median()}, ignore_index=True)

        # Collect informatives on disk usage by observation onto the data column
        memory_usage += int(data[col].memory_usage(index=True, deep=True))
    # Blend of set and list to print the information line as usual
    dtypes_string = ''
    for x in dtypes_uniques:
        dtypes_string += '{}({}), '.format(x, dtypes_listes.count(x))
    print('\ndtypes: {}'.format(dtypes_string))
    # Digit format to write mem usage in comprehensive format
    print('\nmemory usage: {:.4} MB\n'.format(memory_usage / (1024 * 1024)))
    print('\nNombre de lignes: {}\n'.format(len(data[data.columns[0]])))
    print('\nNombre de valeurs non-nulles: {}\n'.format(nb_values_not_null))
    print('\nNombre de valeurs nulles: {}\n'.format(nb_values_is_null))
    print('\nNombre de valeurs au-dessus du seuil: {}\n'.format(nb_values_to_modify))
    return dataframe


def informations(data):
    """
    Fonction d'affichage des paramètres d'une table.
    :param data: un cadre de données
    :return:
    """
    dataframe = pd.DataFrame({'columns': pd.Series([], dtype=str),
                              'type': pd.Series([], dtype=str),
                              'unique': pd.Series([], dtype=int),
                              'non-null': pd.Series([], dtype=int),
                              'mean': pd.Series([], dtype=float),
                              'std': pd.Series([], dtype=float),
                              'pct_null': pd.Series([], dtype=float),
                              'pct>3iqr': pd.Series([], dtype=float),
                              'max': pd.Series([], dtype=float)
                              })
    format_str = '{:^3}{:<1}{:^42}{:<1}{:^7}{:>1}{:^6}{:<1}{:^8}{:<1}{:^7}{:<1}{:^7}{:<1}{:^8}{:<1}{:^8}{:<1}{:^5}{:<1}'
    # Header
    # print(format_str
    #       .format('#', '|', 'Column', '|', 'Dtype', '|', 'Unique', '|', 'Non-Null', '|', 'Mean', '|', 'Std', '|',
    #               'Pct Null', '|', 'Pct>3IQR', '|', ' Max ', '|'))
    # print(format_str.format('---', '|', '------', '|', '-----', '|', '------', '|', '-------', '|', '----', '|', '---',
    #                         '|', '---', '|', '----', '|', '---', '|'))
    # dtypes information
    dtypes_uniques = set()  # Collection of unique elements
    dtypes_listes = []  # list of complete data columns
    memory_usage = 0  # in MB
    nb_values_to_modify: int = 0
    nb_values_not_null: int = 0
    nb_values_is_null: int = 0
    for it, col in enumerate(data.columns):  # Feed the set and the list
        is_a_str_series = True in map((lambda x: type(x) == str), data[col])
        max_col = str(data[col].max()) if not is_a_str_series else 'None'
        # print(str(data[col].max()))
        out_col = data[col][data[col] > 3 * iqr(data[col])].count()
        neg_col = data[col][data[col] < 0].count()
        nb_values_to_modify += out_col + neg_col
        nb_values_not_null += data[col].count()
        try:
            nb_values_is_null += data[col].isnull().value_counts()[True]
        except KeyError as error:
            nb_values_is_null += 0
        dtypes_listes.append(str(data[col].dtype))
        if str(data[col].dtype) not in dtypes_uniques:
            dtypes_uniques.add(str(data[col].dtype))
        # Table body
        # print(
        #     '{:^3}{:<1}{:<42}{:<1}{:^7}{:>1}{:<6}{:<1}{:<8}{:<1}{:<7.5}{:<1}{:<7.5}{:<1}{:<8.5}{:<1}{:<8.5}{:<1}{:<5.5}{:<1}'.format(
        #         str(it), '|', col, '|',
        #         str(data[col].dtype), '|', str(len(data[col].unique())),
        #         '|', str(data[col].count()), '|',
        #         str(data[col].mean())
        #         if (data[col].dtype in ['int64', 'float64']) else '', '|',
        #         str(data[col].std())
        #         if (data[col].dtype in ['int64', 'float64']) else '', '|',
        #         str(pourcent_of_null(data[col])), '|',
        #         str(pourcent_outside_3iqr(data[col])), '|',
        #         max_col, '|'))
        dataframe = dataframe.append({'columns': col, 'type': data[col].dtype, 'unique': len(data[col].unique()),
                                      'non-null': data[col].count(), 'mean': data[col].mean(), 'std': data[col].std(),
                                      'pct_null': pourcent_of_null(data[col]),
                                      'pct>3iqr': pourcent_outside_3iqr(data[col]),
                                      'max': max_col}, ignore_index=True)
        # Collect informatives on disk usage by observation onto the data column
        memory_usage += int(data[col].memory_usage(index=True, deep=True))
    # Blend of set and list to print the information line as usual
    dtypes_string = ''
    for x in dtypes_uniques:
        dtypes_string += '{}({}), '.format(x, dtypes_listes.count(x))
    print('\ndtypes: {}'.format(dtypes_string))
    # Digit format to write mem usage in comprehensive format
    print('\nmemory usage: {:.4} MB\n'.format(memory_usage / (1024 * 1024)))
    print('\nNombre de lignes: {}\n'.format(len(data[data.columns[0]])))
    print('\nNombre de valeurs non-nulles: {}\n'.format(nb_values_not_null))
    print('\nNombre de valeurs nulles: {}\n'.format(nb_values_is_null))
    print('\nNombre de valeurs aberrantes et atypiques: {}\n'.format(nb_values_to_modify))
    return dataframe


def get_clean_db():
    """
    Met en place le fichier de cadre de données pour travailler sur l'analyse statistiques.
    :return:
    """
    data_base = pd.read_pickle(NOM_FICHIER_PICKLE)
    data_without_times = drop_dt_col(data_base)
    data_confidence = select_iqr_interval(data_without_times)
    data_filtered = select_low_null_value_columns(
        data_confidence)  # Sélectionne que les lignes de nutri-score non-nulles.
    informations(data_filtered)


def pca_fct(Xy: pd.DataFrame, to_drop: list, n_comp: int) -> Tuple[decomposition.PCA, np.ndarray, pd.DataFrame]:
    """
    Fonction utilisant un moyennage autour de zéro pour décomposer les données en composantes principales.
    :param Xy: le jeu de données avec la cible
    :param to_drop: le nom de la cible
    :param n_comp: le nombre de composantes
    :return: le pca, le tableau des données moyennée autour de zéro et les features.
    """
    X = Xy.drop(to_drop, axis=1)
    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)
    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(X_scaled)
    # print(pca.explained_variance_ratio_)
    # print(pca.explained_variance_ratio_.sum())
    return pca, X_scaled, X


def scatter_points(pca, X_scaled: np.ndarray, Xy: pd.DataFrame, target: str, limits: list, comp_1: int, comp_2: int):
    """
    Retourne un graphique de point en couleur qui représente les composantes principales.
    :param X_scaled: np.ndarray Les paramètres moyennés autour de zéro
    :param Xy: pd.DataFrame Le jeu de données
    :param target: string Le nom de la cible
    :param limits: list limites x et y du graphique
    :return: plot
    """
    X_projected = pca.transform(X_scaled)
    figure = plt.figure(figsize=(8, 8))
    # afficher chaque observation
    plt.scatter(X_projected[:, comp_1], X_projected[:, comp_2],
                c=Xy.get(target))

    plt.xlim(limits[0][0], limits[0][1])
    plt.ylim(limits[1][0], limits[1][1])
    plt.colorbar()
    return

def eboulis(Xy_df):
    eboulis_vp = pd.DataFrame(
        {'nombre_de_composantes': pd.Series([], dtype=int), 'cumulee_des_valeurs_propres': pd.Series([], dtype=float)})
    for i in range(1, 16):
        pca, X_scaled, X = pca_fct(Xy_df, ['nutrition-score-fr_100g'], i)
        eboulis_vp = eboulis_vp.append(
            {'nombre_de_composantes': i, 'cumulee_des_valeurs_propres': pca.explained_variance_ratio_.sum()},
            ignore_index=True)
    fig, ax1 = plt.subplots()
    fig.suptitle('Eboulis')
    ax1.set_xlabel('Nombre de composantes')
    ax1.set_ylabel('Valeurs propres en cumulées')
    ax1.plot(eboulis_vp['nombre_de_composantes'], eboulis_vp['cumulee_des_valeurs_propres'], 'o-')
    return


def acp_graph(pca, X, limits: list, comp_1: int, comp_2: int):
    """
    Représentation graphique des composantes principales en 2 dimensions.
    :param comp_2: Taille de la fenêtre des y
    :param comp_1: Taille de la fenêtre des x
    :param X: Data Frame contenant le nom des colonnes
    :param pca: Analyse en Composantes principales
    :param limits: Limites du cadre selon les axes x et y
    :return: plot
    """
    pcs = pca.components_
    figure = plt.figure(figsize=(10, 10))

    for i, (x, y) in enumerate(zip(pcs[comp_1, :], pcs[comp_2, :])):
        # Afficher un segment de l'origine au point (x, y)
        plt.plot([0, x], [0, y], color='k', )
        # Afficher le nom (data.columns[i]) de la performance
        plt.text(x, y, X.columns[i], fontsize='11')

    # Afficher une ligne horizontale y=0
    plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')

    # Afficher une ligne verticale x=0
    plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')

    plt.xlim(limits[0][0], limits[0][1])
    plt.ylim(limits[1][0], limits[1][1])
    return


def fit_to_all_distributions(data):
    """
    Calcul la loi statistique d’un échantillon en parallèle avec la fonction
    get_best_distribution_using_chisquared_test().
    """
    dist_names = ['gennorm']  # ['fatiguelife', 'invgauss', 'johnsonsu', 'johnsonsb', 'lognorm', 'norminvgauss',
    # 'powerlognorm', 'exponweib', 'gennorm', 'genextreme', 'pareto']

    params = {}
    for dist_name in dist_names:
        try:
            dist = getattr(st, dist_name)
            param = dist.fit(data)
            params[dist_name] = param
        except Exception as e:
            print(e, "Error occurred in fitting")
            params[dist_name] = "Error"

    return params


def get_best_distribution_using_chisquared_test(data, params):
    """
    Calcul la meilleure loi statistique d’un échantillon pour fonctionner, utilise la fonction
    fit_to_all_distribution()
    """
    histo, bin_edges = np.histogram(data, bins='auto', density=False)
    observed_values = histo

    dist_names = ['gennorm']  # ['fatiguelife', 'invgauss', 'johnsonsu', 'johnsonsb', 'lognorm', 'norminvgauss',
    # 'powerlognorm', 'exponweib', 'gennorm', 'genextreme', 'pareto']

    dist_results = []

    for dist_name in dist_names:

        param = params[dist_name]
        if (param != "Error"):
            # Applying the SSE test
            arg = param[:-2]
            loc = param[-2]
            scale = param[-1]
            cdf = getattr(st, dist_name).cdf(bin_edges, loc=loc, scale=scale, *arg)
            expected_values = len(data) * np.diff(cdf)
            test = (observed_values.sum() - expected_values.sum())*100/observed_values.sum()
            if test < 10**(-8):
                c, p = st.chisquare(observed_values, expected_values, ddof=len(param))
                dist_results.append([dist_name, c, p, expected_values])
            else:
                print('La somme des fréquences observées est différente de plus de 10^-8 des fréquences espérées.')
                return


    # select the best fitted distribution
    best_dist, best_c, best_p, best_expected_values, best_observed_values = None, sys.maxsize, 0, [], []

    for item in dist_results:
        name = item[0]
        c = item[1]
        p = item[2]
        if (not math.isnan(c)):
            if (c < best_c):
                best_c = c
                best_dist = name
                best_p = p
                best_expected_values = expected_values
                best_observed_values = observed_values

    # print the name of the best fit and its p value
    best_dist = 'gennorm'  # À effacer si plus d'une distribution à tester
    print("Goodness of fit for distribution: " + str(best_dist))
    print("Chi2 value: " + str(best_c))
    print("p-value: " + str(best_p))
    #    print("Parameters for the best fit: " + str(params[best_dist]))

    return best_dist, best_c, best_p, best_expected_values, best_observed_values, params[best_dist], dist_results


def interactions(x: str, y: str, bins: int):
    data_score = pd.read_csv(NAME_FILE_SCORE_UNIVARIATE_ANALYSIS, sep='\t', index_col=0)
    groups = data_score.groupby([pd.cut(data_score[y], bins), pd.cut(data_score[x], bins)])
    mat = groups.size().unstack()
    sns.heatmap(mat[::-1])
    plt.tight_layout()
    plt.savefig('interactions_' + x + '_' + y + '.png')


def heatmaps(x: str, y: str, bins: int):
    data_score = pd.read_csv(NAME_FILE_SCORE_UNIVARIATE_ANALYSIS, sep='\t', index_col=0)
    groups = data_score.groupby([pd.cut(data_score[y], bins), pd.cut(data_score[x], bins)])
    mat = groups.size().unstack()
    plt.subplot()
    plt.imshow(mat[::-1], norm=matplotlib.colors.Normalize(0,1), interpolation='gaussian', cmap="rainbow")
    plt.title(x + ' et ' + y, fontdict={'fontsize': '15'})
    plt.tight_layout()
    plt.savefig('imshow_' + x + '_' + y + '.png')
    plt.close()


def barline_plot(bins: int):
    data_score = pd.read_csv(NAME_FILE_SCORE_UNIVARIATE_ANALYSIS, sep='\t', index_col=0)
    columns: list = list(data_score.columns)
    columns.pop()

    for col in columns:
        bins_intervals = pd.cut(data_score[col], bins)
        dataframe_barline_plot = pd.DataFrame(columns={'nutri-score': pd.Series([], dtype=float),
                                                       'densite ' + col.split('_100g')[0]: pd.Series([], dtype=float),
                                                       'observation': pd.Series([], dtype=float),
                                                       'valeurs': pd.Series([], dtype=float)})
        d = dataframe_barline_plot
        cut_nb = bins_intervals.unique().shape[0]
        for i in range(cut_nb):
            d = d.append({'nutri-score': round(
                              data_score['nutrition-score-fr_100g'].loc[
                                  (data_score[col] > bins_intervals.unique()[i].left)
                                  & (data_score[col] < bins_intervals.unique()[
                                      i].right)].mean(), 2),
                          'observation': data_score[col].loc[
                              (data_score[col] > bins_intervals.unique()[i].left)
                              & (data_score[col] < bins_intervals.unique()[i].right)].count(),
                          'valeurs': round(
                              data_score[col].loc[
                                  (data_score[col] > bins_intervals.unique()[i].left)
                                  & (data_score[col] < bins_intervals.unique()[
                                      i].right)].mean(), 2)},
                         ignore_index=True)
        d['densite ' + col.split('_100g')[0]] = d['observation'] / d['observation'].sum() * 100
        d = d.sort_values(by=['valeurs']).reset_index()
        matplotlib.rc_file_defaults()
        ax1 = sns.set_style(style=None, rc=None)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        sns.lineplot(data=d['nutri-score'], marker='o', sort=False, ax=ax1)
        ax2 = ax1.twinx()


        plot = sns.barplot(data=d, x='valeurs', y='densite ' + col.split('_100g')[0], color='orange', alpha=0.5, ax=ax2)
        plot.set_title('Moyenne du ' + 'nutri-score' + ' par ' + col.split('_100g')[0])
        ax1.set(xlabel='quantité ' + col.split('_100g')[0])
        plt.savefig('nutriscore_par_' + col + '.png')



def main():
    # repair_db()
    # get_clean_db()
    barline_plot(10)


if __name__ == '__main__':
    main()
