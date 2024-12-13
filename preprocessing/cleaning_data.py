

"""
This script provides functions and utilities to preprocess data and normalize input for a machine learning model 
deployed via Streamlit. It loads reference data, defines categorical mappings, and applies normalization using 
MinMaxScaler.

Key Components:
---------------
1. normalize_data(data, reference_data):
   - Function to normalize input data based on a reference dataset using MinMaxScaler.

2. preprocess():
   - Function to load reference data, drop unnecessary columns, and define mappings for categorical variables.

Modules Used:
-------------
- streamlit: For building interactive web applications.
- pandas: For handling data operations.
- numpy: For array manipulations.
- joblib: For loading pre-trained models.
- sklearn.preprocessing.MinMaxScaler: For feature scaling.

Usage:
------
This script is intended to be used as part of a Streamlit web application. The `preprocess` function loads 
the reference dataset, applies the necessary transformations, and returns data suitable for inference 
or further processing.
"""



import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler





#Function to normalize input data


class Cleaning :


    def __init__(self):
         pass



    """
    A class to handle data cleaning, normalization, and preprocessing for real estate prediction data.

    Methods:
    - normalize_data: Normalizes numerical data using MinMaxScaler.
    - preprocess: Loads reference data, applies preprocessing steps, and provides mappings for encoded columns.
    """
    
    

    def normalize_data(self,data, reference_data):
        """
        Normalizes the input data based on the provided reference dataset.

        Args:
        - data (DataFrame or array-like): The data to be normalized.
        - reference_data (DataFrame): The reference data used to fit the scaler.

        Returns:
        - ndarray: The normalized data.
        """
        scaler = MinMaxScaler()
        scaler.fit(reference_data)  # Fit the scaler to reference data
        return scaler.transform(data)



    def preprocess(self):
        """
        Preprocesses real estate data for machine learning predictions.

        Steps:
        1. Loads a reference dataset for scaling.
        2. Drops unnecessary columns like 'Price' and 'Id'.
        3. Defines mappings for encoded categorical data.
        4. Ensures data is in the correct format for model predictions.

        Returns:
        - dict: Mappings for categorical columns.
        """

        reference_data = pd.read_csv("/home/learner/Desktop/Deplyment/Immoliza_app/preprocessing/ED.csv", index_col=0).drop(columns=["Price", "Id"])

        # Define mappings for categorical columns
        mappings = {
            "State": {
                1: "Good",
                2: "Not Known",
                3: "As new",
                4: "To renovate",
                5: "To be done up",
                6: "Just renovated",
                7: "To restore",
            },
            #"Type_encoded": {0: "Apartment", 1: "House"},
            "SubType_encoded": {
                0: "house",
                1: "villa",
                2: "mixed-use-building",
                3: "apartment",
                4: "exceptional-property",
                5: "flat-studio",
                6: "duplex",
                7: "ground-floor",
                8: "penthouse",
                9: "mansion",
                10: "apartment-block",
                11: "town-house",
            }

            ,

            "Locality_encoded":{0: 'Ertvelde', 1: 'Hornu', 2: 'Beersel', 3: 'Geraardsbergen', 4: 'Jumet', 5: 'Blaasveld', 6: 'Herentals', 7: 'Beerlegem', 8: 'Aalst', 9: 'Gistel', 10: 'Herzele', 11: 'Leuven', 12: 'Saint-Amand', 13: 'Grobbendonk', 14: 'Deerlijk', 15: 'Brugge', 16: 'Kemzeke', 17: 'Berlare', 18: 'Haacht', 19: 'Gellik', 20: 'Mol', 21: 'Wilsele', 22: 'Fontaine-Valmont', 23: 'Emelgem', 24: 'Edegem', 25: 'Messelbroek', 26: 'Antwerpen', 27: 'Lauwe', 28: 'Boekhoute', 29: 'Kessel-Lo', 30: 'Asper', 31: 'Anderlecht', 32: 'Afsnee', 33: 'Vorst', 34: 'Lembeek', 35: 'Aartselaar', 36: 'Bassevelde', 37: 'Eeklo', 38: 'Marcinelle', 39: 'Courcelles', 40: 'Haine-Saint-Paul', 41: 'Gent', 42: 'Schellebelle', 43: 'Sterrebeek', 44: 'Kortenberg', 45: 'Auby-Sur-Semois', 46: 'Berchem', 47: 'Anseremme', 48: 'Asse', 49: 'Peutie', 50: 'Aarschot', 51: 'Sint-pieters-woluwe', 52: 'Baisy-Thy', 53: 'Beerzel', 54: 'Keerbergen', 55: 'Ronse', 56: 'Kapellen', 57: 'Schaarbeek', 58: 'Bras', 59: 'Achêne', 60: 'Nil-Saint-Vincent-Saint-Martin', 61: 'Geluwe', 62: 'Pommeroeul', 63: 'Koekelberg', 64: 'Beveren-waas', 65: 'Blankenberge', 66: 'Ganshoren', 67: 'Assebroek', 68: 'Sint-Andries', 69: 'Damme', 70: 'Maldegem', 71: 'Rillaar', 72: 'Astene', 73: 'Deurle', 74: 'Sint-martens-latem', 75: 'Quaregnon', 76: 'Gontrode', 77: 'Elsene', 78: 'Faymonville', 79: 'Leopoldsburg', 80: 'Bressoux', 81: 'Wemmel', 82: 'Laken', 83: 'Wilrijk', 84: 'Casteau', 85: 'Burcht', 86: 'Deurne', 87: 'Bierghes', 88: 'Acoz', 89: 'Bassilly', 90: 'Dongelberg', 91: 'Enines', 92: 'Amougies', 93: 'Kerksken', 94: 'Bonheiden', 95: 'Brielen', 96: 'Beveren', 97: 'Harelbeke', 98: 'Desselgem', 99: 'Bertem', 100: 'Awirs', 101: 'Herselt', 102: 'Herenthout', 103: 'Boezinge', 104: 'Heusy', 105: 'La Reid', 106: 'Borgerhout', 107: 'Aisemont', 108: 'Schriek', 109: 'Loppem', 110: 'Bovekerke', 111: 'Adinkerke', 112: 'Grandrieu', 113: 'Wijnegem', 114: 'Bouffioulx', 115: 'Knokke', 116: 'Dorinne', 117: 'Saint-ghislain', 118: 'Turnhout', 119: 'Nieuwpoort', 120: 'Nieuwrode', 121: 'Gouy-Lez-Piéton', 122: 'Horrues', 123: 'Mariakerke', 124: 'De Haan', 125: 'Nieuwkerken-Waas', 126: 'Kortrijk', 127: 'Welkenraedt', 128: 'Court-saint-etienne', 129: 'Chaineux', 130: 'Harsin', 131: 'Barvaux-Sur-Ourthe', 132: 'Groot-Bijgaarden', 133: 'Sint-agatha-berchem', 134: 'Kapelle-op-den-bos', 135: 'Ans', 136: 'Beez', 137: 'Luingne', 138: 'Malle', 139: 'Lommel', 140: 'Liedekerke', 141: 'Merksem', 142: 'Bazel', 143: 'Loenhout', 144: 'Mons', 145: 'Neerwaasten', 146: 'Eupen', 147: 'Brussel', 148: 'Charleroi', 149: 'Archennes', 150: 'Berg', 151: 'Lodelinsart', 152: 'Hasselt', 153: 'Oostende', 154: 'Glain', 155: 'Herent', 156: 'Gooik', 157: 'Berneau', 158: 'Leers-Et-Fosteau', 159: 'Hoboken', 160: 'S Herenelderen', 161: 'Limal', 162: 'Bassenge', 163: 'De Klinge', 164: 'Wondelgem', 165: 'Affligem', 166: 'De Pinte', 167: 'Ave-Et-Auffe', 168: 'Sint-Joost-ten-Node', 169: 'Bunsbeek', 170: 'Bost', 171: 'Oevel', 172: 'Sint-Amandsberg', 173: 'Sint-gillis', 174: 'Machelen', 175: 'Arendonk', 176: 'Bogaarden', 177: 'Boussoit', 178: 'Waterloo', 179: 'Bevere', 180: 'Booischot', 181: 'Ukkel', 182: 'Eke', 183: 'Goutroux', 184: 'Sint-lambrechts-woluwe', 185: 'Arlon', 186: 'Edingen', 187: 'Moerbeke-waas', 188: 'Hever', 189: 'Avelgem', 190: 'Brecht', 191: 'Ben-Ahin', 192: 'Oudergem', 193: 'Sint-katelijne-waver', 194: 'Boncelles', 195: 'Anzegem', 196: 'Retie', 197: 'Hamme', 198: 'Mortsel', 199: 'Hoevenen', 200: 'Beverst', 201: 'Gentbrugge', 202: 'Mechelen', 203: 'Grivegnee', 204: 'Bettincourt', 205: 'Vosselaar', 206: 'Evere', 207: 'Boom', 208: 'Montegnée', 209: 'Sint-Martens-Lierde', 210: 'Attert', 211: 'Hondelange', 212: 'Aubange', 213: 'Musson', 214: 'Athus', 215: 'Etalle', 216: 'Châtillon', 217: "Bersillies-L'Abbaye", 218: 'Houtvenne', 219: 'Anloy', 220: 'Roux', 221: 'Gullegem', 222: 'Sars-La-Buissière', 223: 'Halle', 224: 'Baisieux', 225: 'Jette', 226: 'Eindhout', 227: 'Knesselare', 228: 'Heist-Aan-Zee', 229: 'Middelkerke', 230: 'Lippelo', 231: 'Sint-jans-molenbeek', 232: 'Helchteren', 233: 'Bullange', 234: 'Assenede', 235: 'Massemen', 236: 'Kalken', 237: 'Zomergem', 238: 'Zonhoven', 239: 'Avekapelle', 240: 'Bernissart', 241: 'Ensival', 242: 'Dolembreux', 243: 'Itegem', 244: 'Hastière-Lavaux', 245: 'Melsbroek', 246: 'Clabecq', 247: 'Elene', 248: 'Aalter', 249: 'Herstal', 250: 'Ingelmunster', 251: 'Koksijde', 252: 'Hertsberge', 253: 'Berlaar', 254: 'Habay-La-Neuve', 255: 'Vance', 256: 'Dadizele', 257: 'Heusden', 258: 'Handzame', 259: 'Deux-Acren', 260: 'Appelterre-Eichem', 261: 'Ettelgem', 262: 'Awans', 263: 'Beerst', 264: 'Fleurus', 265: 'Ledegem', 266: 'Bléharies', 267: 'Rixensart', 268: 'Etterbeek', 269: 'Ath', 270: 'Koningshooikt', 271: 'Balen', 272: 'Aywaille', 273: 'Aaigem', 274: 'Monstreux', 275: 'Brussegem', 276: 'Blaugies', 277: 'Chastre-Villeroux-Blanmont', 278: 'Hallaar', 279: 'Mazenzele', 280: 'Balâtre', 281: 'Champion', 282: 'Braine-le-comte', 283: 'Dave', 284: 'Beerse', 285: 'Andenne', 286: 'Koersel', 287: 'Rocherath', 288: 'Tessenderlo', 289: 'Bavikhove', 290: 'Zwevegem', 291: 'Impe', 292: 'Neerpelt', 293: 'Emines', 294: 'Wommelgem', 295: 'Nossegem', 296: 'Grimminge', 297: 'Hoeselt', 298: 'Erembodegem', 299: 'Lombardsijde', 300: 'Bierset', 301: 'Aye', 302: 'Beaufays', 303: 'Chaudfontaine', 304: 'Broechem', 305: 'Begijnendijk', 306: 'Niel', 307: 'Baronville', 308: 'S Gravenwezel', 309: 'Houwaart', 310: 'Braine-le-château', 311: "Braine-l'alleud", 312: 'Beringen', 313: 'Drongen', 314: 'Schoten', 315: 'Ekeren', 316: 'Borsbeek', 317: 'Balegem', 318: 'Ternat', 319: 'Zele', 320: 'Meerhout', 321: 'Angre', 322: 'Huise', 323: 'Carnières', 324: 'Chevron', 325: 'Brasschaat', 326: 'Baudour', 327: 'Abolens', 328: 'Gottignies', 329: 'Bailleul', 330: 'Wolvertem', 331: 'Jemeppe-Sur-Meuse', 332: 'Essen', 333: 'Boechout', 334: 'Paal', 335: 'Beernem', 336: 'Drieslinter', 337: 'Amberloup', 338: 'Sint-genesius-rode', 339: 'Relegem', 340: 'Daknam', 341: 'Cornesse', 342: 'Colfontaine', 343: 'Oostakker', 344: 'Chapelle-lez-herlaimont', 345: 'Lessines', 346: 'Huizingen', 347: 'Hastière-Par-Delà', 348: 'Gaasbeek', 349: 'Wingene', 350: 'Strombeek-Bever', 351: 'Ransart', 352: 'Kaprijke', 353: 'Lembeke', 354: 'Arsimont', 355: 'Beselare', 356: 'Lillois-Witterzée', 357: 'Eliksem', 358: 'Aspelare', 359: 'Waarschoot', 360: 'Hoogstraten', 361: 'Assent', 362: 'Baal', 363: 'Waregem', 364: 'Geetbets', 365: 'Kaster', 366: 'Aartrijke', 367: 'Onze-Lieve-Vrouw-Waver', 368: 'Dampicourt', 369: 'Halen', 370: 'Holsbeek', 371: 'Neder-Over-Heembeek', 372: 'Arbre', 373: 'Couillet', 374: 'Bottelare', 375: 'Denderbelle', 376: 'Lot', 377: 'Oudenaken', 378: 'Ruisbroek', 379: 'Linkhout', 380: 'Biesme', 381: 'Londerzeel', 382: 'Vieux-Genappe', 383: 'Angleur', 384: 'Kuurne', 385: 'Denderhoutem', 386: 'Bekegem', 387: 'Bommershoven', 388: 'Ere', 389: 'Linden', 390: 'Bornem', 391: 'Dudzele', 392: 'Waasmunster', 393: 'Oostnieuwkerke', 394: 'Berbroek', 395: 'Aarsele', 396: 'Beuzet', 397: 'Assenois', 398: 'Hansbeke', 399: 'Alken', 400: 'Arville', 401: 'Florenville', 402: 'Meulebeke', 403: 'Zutendaal', 404: 'Overpelt', 405: 'Appels', 406: 'Autelbas', 407: 'Trazegnies', 408: 'Saint-georges-sur-meuse', 409: 'Fize-Fontaine', 410: 'Corbais', 411: 'Linkebeek', 412: 'Havre', 413: 'Geel', 414: 'Maaseik', 415: 'Heverlee', 416: 'Bornival', 417: 'Chênee', 418: 'Nederzwalm-Hermelgem', 419: 'Jabbeke', 420: 'Comblain-au-pont', 421: 'Glimes', 422: 'Wachtebeke', 423: 'Kontich', 424: 'Ellemelle', 425: 'Bastogne', 426: 'Bierges', 427: 'Battignies', 428: 'Werchter', 429: 'Rotselaar', 430: 'Boortmeerbeek', 431: 'Gilly', 432: 'Lendelede', 433: 'Ghlin', 434: 'Houthulst', 435: 'Breendonk', 436: 'Eppegem', 437: 'Bierwart', 438: 'Destelbergen', 439: 'Bevel', 440: 'Wezembeek-oppem', 441: 'Duffel', 442: 'Hombeek', 443: 'Ougrée', 444: 'Baardegem', 445: 'Sint-laureins', 446: 'Agimont', 447: 'Dilsen', 448: 'Hoegaarden', 449: 'Boignée', 450: 'Arbrefontaine', 451: 'Dourbes', 452: 'Buggenhout', 453: 'Obourg', 454: 'Grote-Brogel', 455: 'Forêt', 456: 'Marbais', 457: 'Achel', 458: 'Bocholt', 459: 'Embourg', 460: 'Sint-Eloois-Vijve', 461: 'Alsemberg', 462: 'Gemmenich', 463: 'Adegem', 464: 'Gits', 465: 'Bellegem', 466: 'Torhout', 467: 'Donceel', 468: 'Bredene', 469: 'Malèves-Sainte-Marie-Wastines', 470: 'Outer', 471: 'Wanfercée-Baulet', 472: 'Eugies', 473: 'Dampremy', 474: 'Schelle', 475: 'Hemiksem', 476: 'Ayeneux', 477: 'Aublain', 478: 'Averbode', 479: 'Otegem', 480: 'Amay', 481: 'Hanzinelle', 482: 'Egem', 483: 'Lo', 484: 'Antoing', 485: 'Genk', 486: 'Louvain-La-Neuve', 487: 'Hautrage', 488: 'Tremelo', 489: 'Berlingen', 490: 'Kermt', 491: 'Bissegem', 492: 'Galmaarden', 493: 'Sint-Kornelis-Horebeke', 494: 'Ruiselede', 495: 'Eisden', 496: 'Ellikom', 497: 'Haasrode', 498: 'Buissenal', 499: 'Berloz', 500: 'Baugnies', 501: 'Antheit', 502: 'Aiseau', 503: 'Houffalize', 504: 'Courrière', 505: 'Bikschote', 506: 'Mont-Sur-Marchienne', 507: 'Beervelde', 508: 'Oostrozebeke', 509: 'Reet', 510: 'Brakel', 511: 'Watermaal-bosvoorde', 512: 'Croix-Lez-Rouveroy', 513: 'Ottignies', 514: 'Farciennes', 515: 'Herseaux', 516: 'Basècles', 517: 'Houtain-Le-Val', 518: 'Alveringem', 519: 'Ittre', 520: 'Nieuwkerke', 521: 'Hove', 522: 'Bourseigne-Neuve', 523: 'Spa', 524: 'Elversele', 525: 'Overijse', 526: 'Ooigem', 527: 'Diepenbeek', 528: 'Chièvres', 529: 'Erbaut', 530: 'Bauffe', 531: 'Wijgmaal', 532: 'Flénu', 533: 'Zwijnaarde', 534: 'Montigny-le-tilleul', 535: 'Castillon', 536: 'Houtaing', 537: 'Aubel', 538: 'Andrimont', 539: 'Dison', 540: 'Pulle', 541: 'Desteldonk', 542: 'Dessel', 543: 'Velm', 544: 'Bossuit', 545: 'Bure', 546: 'Vloesberg', 547: 'Genval', 548: 'Oud-turnhout', 549: 'Buvrinnes', 550: 'Helkijn', 551: "Fontaine-l'evêque", 552: 'Kain', 553: 'Fosse', 554: 'Stembert', 555: 'Poperinge', 556: 'Sint-Stevens-Woluwe', 557: 'Gijzegem', 558: 'Waterland-Oudeman', 559: 'Bodegnée', 560: 'Grimbergen', 561: 'Rijkevorsel', 562: 'Vorselaar', 563: 'Nandrin', 564: 'Moen', 565: 'Diegem', 566: 'Ferrières', 567: 'Kasterlee', 568: 'Anhée', 569: 'Ehein', 570: 'Lanaye', 571: 'Milmort', 572: 'Drogenbos', 573: 'Buzet', 574: 'Denderleeuw', 575: 'Rekem', 576: 'Krombeke', 577: 'Binkom', 578: 'Bihain', 579: 'Zelzate', 580: 'Belsele', 581: 'Morlanwelz-Mariemont', 582: 'Beffe', 583: 'Bergilers', 584: 'Doel', 585: 'Bellefontaine', 586: "Ecaussinnes-D'Enghien", 587: 'Boutersem', 588: 'Tertre', 589: 'Ghislenghien', 590: 'Dilbeek', 591: 'Beek', 592: 'Pellenberg', 593: 'Kortessem', 594: 'Haren', 595: 'Massenhoven', 596: 'Komen', 597: 'Halanzy', 598: 'Hermée', 599: 'Kessenich', 600: 'Beausaint', 601: 'Baillonville', 602: 'Heure-Le-Romain', 603: 'Maisières', 604: 'Bever', 605: 'Gosselies', 606: 'Olmen', 607: 'Kwaadmechelen', 608: 'Fagnolle', 609: 'Oostvleteren', 610: 'Liers', 611: 'Henri-Chapelle', 612: 'Lobbes', 613: 'Deftinge', 614: 'Haut-Ittre', 615: 'Dworp', 616: 'Péruwelz', 617: 'Genoelselderen', 618: 'Gouvy', 619: 'Bellecourt', 620: 'Boekhout', 621: 'Meerle', 622: 'Attre', 623: 'Boussu', 624: 'Berzée', 625: 'Scherpenheuvel', 626: 'Bailièvre', 627: 'Petit-Thier', 628: 'Argenteau', 629: 'Clermont', 630: 'Floreffe', 631: 'Aische-En-Refail', 632: 'Corenne', 633: 'Bellem', 634: 'Bavegem', 635: 'Barbençon', 636: 'Wijtschate', 637: 'Fexhe-Slins', 638: 'Vaux-Sous-Chèvremont', 639: 'Recht', 640: 'Saint-Servais', 641: 'Vlezenbeek', 642: 'Cherain', 643: 'La Hulpe', 644: 'Bois-De-Lessines', 645: 'Dentergem', 646: 'Ciergnon', 647: 'Marchin', 648: 'Comblain-Fairon', 649: 'Linsmeau', 650: 'Cour-Sur-Heure', 651: 'Fléron', 652: 'Arquennes', 653: 'Avennes', 654: 'Givry', 655: 'Aubechies', 656: 'Blaton', 657: 'Chaussée-Notre-Dame-Louvignies', 658: 'Carlsbourg', 659: 'Alle', 660: 'Lichtervelde', 661: 'Clermont-Sous-Huy', 662: 'Blanden', 663: 'Beauvechain', 664: 'Kersbeek-Miskom', 665: 'Langdorp', 666: 'Baillamont', 667: 'Meise', 668: 'Bossière', 669: 'Beert', 670: 'Opglabbeek', 671: 'Bellevaux-Ligneuville', 672: 'Lint', 673: 'Audregnies', 674: 'Hainin', 675: 'Weelde', 676: 'Bouillon', 677: 'Vivegnis', 678: 'Hompré', 679: 'Stave', 680: 'Hognoul', 681: 'Habay-La-Vieille', 682: 'Sohier', 683: 'Bonlez', 684: 'Anderlues', 685: 'Neufmaison', 686: 'Celles', 687: 'Limelette', 688: 'Ellezelles', 689: 'Kruishoutem', 690: 'Alleur', 691: 'Ucimont', 692: 'Couture-Saint-Germain', 693: 'Elsegem', 694: 'Juprelle', 695: 'Seneffe', 696: 'Bon-Secours', 697: 'Doische', 698: 'Lamine', 699: 'Epinois', 700: 'Ramegnies-Chin', 701: 'Fronville', 702: 'Amonines', 703: 'Bleid', 704: 'Habergy', 705: 'Gierle', 706: 'Martelange', 707: 'Battice', 708: 'Duisburg', 709: 'Mannekensvere', 710: 'Ravels', 711: 'Onze-Lieve-Vrouw-Lombeek', 712: 'Frasnes-Lez-Gosselies', 713: 'Abée', 714: 'Heffen', 715: 'Monceau-Sur-Sambre', 716: 'Haillot', 717: 'Welle', 718: 'Neufchâteau', 719: 'Autre-Eglise', 720: 'Neuville-En-Condroz', 721: 'Houdemont', 722: 'Rosières', 723: 'Muizen', 724: 'Asquillies', 725: 'Jalhay', 726: 'Les Bulles', 727: 'Loker', 728: 'Huldenberg', 729: 'Herfelingen', 730: 'Hour', 731: 'Chiny', 732: 'Gelbressée', 733: 'Modave', 734: 'Kerkhove', 735: 'Grandvoir', 736: 'Crisnée', 737: 'Cuesmes', 738: 'Ardooie', 739: 'Dottenijs', 740: 'Kraainem', 741: 'Elewijt', 742: 'Outrijve', 743: 'Vremde', 744: 'Lovendegem', 745: 'Tenneville', 746: 'Middelburg', 747: 'Everberg', 748: 'Barchon', 749: 'Héron', 750: 'Sorinnes', 751: 'Maissin', 752: 'Brugelette', 753: 'Esneux', 754: 'Hermalle-Sous-Argenteau', 755: 'Baileux', 756: 'Momignies', 757: 'Olne', 758: 'Bende', 759: 'Waudrez', 760: 'Leffinge', 761: 'Chanly', 762: 'Vinderhoute', 763: 'Céroux-Mousty', 764: 'Dergneau', 765: 'Hodeige', 766: 'Corbion', 767: 'Saint-Remy', 768: 'Bierbeek', 769: 'Haccourt', 770: 'Heer', 771: 'Amblève', 772: 'Iddergem', 773: 'Belgrade', 774: 'Basse-Bodeux', 775: 'Borchtlombeek', 776: 'Romsée', 777: 'Buizingen', 778: 'Itterbeek', 779: 'Montleban', 780: 'Cerfontaine', 781: 'Poulseur', 782: 'Lisogne', 783: 'Minderhout', 784: 'Hoeilaart', 785: 'Pollare', 786: 'Olen', 787: 'Gages', 788: 'Dochamps', 789: 'Sint-Joris-Weert', 790: 'Borsbeke', 791: 'Anthisnes', 792: 'Etikhove', 793: 'Varendonk', 794: 'Melen', 795: 'Montignies-Sur-Sambre', 796: 'Heestert', 797: 'Francorchamps', 798: 'Sinaai-Waas', 799: 'Bertogne', 800: 'Fauvillers', 801: 'Sélange', 802: 'Papignies', 803: 'Plainevaux', 804: 'Saint-Mard', 805: 'Aalbeke'},
        }


        # Reverse mappings for user-friendly input
        reverse_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in mappings.items()}

        return reverse_mappings,reference_data,mappings


