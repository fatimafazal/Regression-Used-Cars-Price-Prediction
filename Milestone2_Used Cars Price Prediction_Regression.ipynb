{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xWKrvyhfVL3e"
   },
   "source": [
    "# **Milestone 2**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Rzh8pcvmtOS8"
   },
   "source": [
    "## **Model Building**\n",
    "\n",
    "1. What we want to predict is the \"Price\". We will use the normalized version 'price_log' for modeling.\n",
    "2. Before we proceed to the model, we'll have to encode categorical features. We will drop categorical features like Name. \n",
    "3. We'll split the data into train and test, to be able to evaluate the model that we build on the train data.\n",
    "4. Build Regression models using train data.\n",
    "5. Evaluate the model performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 308,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Libraries to help with reading and manipulating data\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Library to split data\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Libraries to help with data visualization\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Removes the limit for the number of displayed columns\n",
    "pd.set_option(\"display.max_columns\", None)\n",
    "# Sets the limit for the number of displayed rows\n",
    "pd.set_option(\"display.max_rows\", 100)\n",
    "\n",
    "# Import libraries for building linear regression model\n",
    "import statsmodels.api as sm\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn import metrics\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Note:** Please load the data frame that was saved in Milestone 1 here before separating the data, and then proceed to the next step in Milestone 2."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Load the data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 309,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              Year  Kilometers_Driven      Mileage       Engine        Power  \\\n",
      "count  6018.000000        6018.000000  6018.000000  6018.000000  6018.000000   \n",
      "mean   2013.357594       57668.047690    18.135329  1620.318877   112.605982   \n",
      "std       3.269677       37878.783175     4.581823   600.052752    53.552585   \n",
      "min    1998.000000         171.000000     0.000000    72.000000    34.200000   \n",
      "25%    2011.000000       34000.000000    15.170000  1198.000000    74.907500   \n",
      "50%    2014.000000       53000.000000    18.160000  1493.000000    93.700000   \n",
      "75%    2016.000000       73000.000000    21.100000  1984.000000   138.100000   \n",
      "max    2019.000000      775000.000000    33.540000  5998.000000   560.000000   \n",
      "\n",
      "             Seats    New_price        Price  kilometers_driven_log  \\\n",
      "count  6018.000000  2234.000000  6018.000000            6018.000000   \n",
      "mean      5.277999    22.969868     9.470243              10.757961   \n",
      "std       0.803837    26.967994    11.165926               0.713022   \n",
      "min       2.000000     3.910000     0.440000               5.141664   \n",
      "25%       5.000000     7.450000     3.500000              10.434116   \n",
      "50%       5.000000    11.295000     5.640000              10.878047   \n",
      "75%       5.000000    26.275000     9.950000              11.198215   \n",
      "max      10.000000   230.000000   160.000000              13.560618   \n",
      "\n",
      "         price_log       CarAge  \n",
      "count  6018.000000  6018.000000  \n",
      "mean      1.824705  2013.357594  \n",
      "std       0.873606     3.269677  \n",
      "min      -0.820981  1998.000000  \n",
      "25%       1.252763  2011.000000  \n",
      "50%       1.729884  2014.000000  \n",
      "75%       2.297573  2016.000000  \n",
      "max       5.075174  2019.000000  \n"
     ]
    }
   ],
   "source": [
    "# Load the data\n",
    "cars_data = pd.read_csv(\"cars_data_updated.csv\")\n",
    "print(cars_data.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 310,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The median mileage is: 18.16\n"
     ]
    }
   ],
   "source": [
    "# Calculate the median of the 'Mileage' column\n",
    "median_mileage = cars_data['Mileage'].median()\n",
    "print(f\"The median mileage is: {median_mileage}\")\n",
    "\n",
    "# Replace mileage values less than 0.1 with the median value\n",
    "cars_data['Mileage'] = np.where(cars_data['Mileage'] < 0.1, median_mileage, cars_data['Mileage'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 311,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Name</th>\n",
       "      <th>Location</th>\n",
       "      <th>Year</th>\n",
       "      <th>Kilometers_Driven</th>\n",
       "      <th>Fuel_Type</th>\n",
       "      <th>Transmission</th>\n",
       "      <th>Owner_Type</th>\n",
       "      <th>Mileage</th>\n",
       "      <th>Engine</th>\n",
       "      <th>Power</th>\n",
       "      <th>Seats</th>\n",
       "      <th>New_price</th>\n",
       "      <th>Price</th>\n",
       "      <th>kilometers_driven_log</th>\n",
       "      <th>price_log</th>\n",
       "      <th>CarAge</th>\n",
       "      <th>Brand</th>\n",
       "      <th>Model</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Maruti Wagon R LXI CNG</td>\n",
       "      <td>Mumbai</td>\n",
       "      <td>2010</td>\n",
       "      <td>72000</td>\n",
       "      <td>CNG</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First</td>\n",
       "      <td>26.60</td>\n",
       "      <td>998.0</td>\n",
       "      <td>58.16</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.510</td>\n",
       "      <td>1.75</td>\n",
       "      <td>11.184421</td>\n",
       "      <td>0.559616</td>\n",
       "      <td>2010</td>\n",
       "      <td>MARUTI</td>\n",
       "      <td>WagonR</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Hyundai Creta 1.6 CRDi SX Option</td>\n",
       "      <td>Pune</td>\n",
       "      <td>2015</td>\n",
       "      <td>41000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First</td>\n",
       "      <td>19.67</td>\n",
       "      <td>1582.0</td>\n",
       "      <td>126.20</td>\n",
       "      <td>5.0</td>\n",
       "      <td>16.060</td>\n",
       "      <td>12.50</td>\n",
       "      <td>10.621327</td>\n",
       "      <td>2.525729</td>\n",
       "      <td>2015</td>\n",
       "      <td>HYUNDAI</td>\n",
       "      <td>Creta1.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Honda Jazz V</td>\n",
       "      <td>Chennai</td>\n",
       "      <td>2011</td>\n",
       "      <td>46000</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First</td>\n",
       "      <td>18.20</td>\n",
       "      <td>1199.0</td>\n",
       "      <td>88.70</td>\n",
       "      <td>5.0</td>\n",
       "      <td>8.610</td>\n",
       "      <td>4.50</td>\n",
       "      <td>10.736397</td>\n",
       "      <td>1.504077</td>\n",
       "      <td>2011</td>\n",
       "      <td>HONDA</td>\n",
       "      <td>JazzV</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Maruti Ertiga VDI</td>\n",
       "      <td>Chennai</td>\n",
       "      <td>2012</td>\n",
       "      <td>87000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First</td>\n",
       "      <td>20.77</td>\n",
       "      <td>1248.0</td>\n",
       "      <td>88.76</td>\n",
       "      <td>7.0</td>\n",
       "      <td>11.215</td>\n",
       "      <td>6.00</td>\n",
       "      <td>11.373663</td>\n",
       "      <td>1.791759</td>\n",
       "      <td>2012</td>\n",
       "      <td>MARUTI</td>\n",
       "      <td>ErtigaVDI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Audi A4 New 2.0 TDI Multitronic</td>\n",
       "      <td>Coimbatore</td>\n",
       "      <td>2013</td>\n",
       "      <td>40670</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Automatic</td>\n",
       "      <td>Second</td>\n",
       "      <td>15.20</td>\n",
       "      <td>1968.0</td>\n",
       "      <td>140.80</td>\n",
       "      <td>5.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>17.74</td>\n",
       "      <td>10.613246</td>\n",
       "      <td>2.875822</td>\n",
       "      <td>2013</td>\n",
       "      <td>AUDI</td>\n",
       "      <td>A4New</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6013</th>\n",
       "      <td>Maruti Swift VDI</td>\n",
       "      <td>Delhi</td>\n",
       "      <td>2014</td>\n",
       "      <td>27365</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First</td>\n",
       "      <td>28.40</td>\n",
       "      <td>1248.0</td>\n",
       "      <td>74.00</td>\n",
       "      <td>5.0</td>\n",
       "      <td>7.880</td>\n",
       "      <td>4.75</td>\n",
       "      <td>10.217020</td>\n",
       "      <td>1.558145</td>\n",
       "      <td>2014</td>\n",
       "      <td>MARUTI</td>\n",
       "      <td>SwiftVDI</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6014</th>\n",
       "      <td>Hyundai Xcent 1.1 CRDi S</td>\n",
       "      <td>Jaipur</td>\n",
       "      <td>2015</td>\n",
       "      <td>100000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First</td>\n",
       "      <td>24.40</td>\n",
       "      <td>1120.0</td>\n",
       "      <td>71.00</td>\n",
       "      <td>5.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.00</td>\n",
       "      <td>11.512925</td>\n",
       "      <td>1.386294</td>\n",
       "      <td>2015</td>\n",
       "      <td>HYUNDAI</td>\n",
       "      <td>Xcent1.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6015</th>\n",
       "      <td>Mahindra Xylo D4 BSIV</td>\n",
       "      <td>Jaipur</td>\n",
       "      <td>2012</td>\n",
       "      <td>55000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Second</td>\n",
       "      <td>14.00</td>\n",
       "      <td>2498.0</td>\n",
       "      <td>112.00</td>\n",
       "      <td>8.0</td>\n",
       "      <td>11.690</td>\n",
       "      <td>2.90</td>\n",
       "      <td>10.915088</td>\n",
       "      <td>1.064711</td>\n",
       "      <td>2012</td>\n",
       "      <td>MAHINDRA</td>\n",
       "      <td>XyloD4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6016</th>\n",
       "      <td>Maruti Wagon R VXI</td>\n",
       "      <td>Kolkata</td>\n",
       "      <td>2013</td>\n",
       "      <td>46000</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First</td>\n",
       "      <td>18.90</td>\n",
       "      <td>998.0</td>\n",
       "      <td>67.10</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.510</td>\n",
       "      <td>2.65</td>\n",
       "      <td>10.736397</td>\n",
       "      <td>0.974560</td>\n",
       "      <td>2013</td>\n",
       "      <td>MARUTI</td>\n",
       "      <td>WagonR</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6017</th>\n",
       "      <td>Chevrolet Beat Diesel</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>2011</td>\n",
       "      <td>47000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First</td>\n",
       "      <td>25.44</td>\n",
       "      <td>936.0</td>\n",
       "      <td>57.60</td>\n",
       "      <td>5.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.50</td>\n",
       "      <td>10.757903</td>\n",
       "      <td>0.916291</td>\n",
       "      <td>2011</td>\n",
       "      <td>CHEVROLET</td>\n",
       "      <td>BeatDiesel</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>6018 rows × 18 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                  Name    Location  Year  Kilometers_Driven  \\\n",
       "0               Maruti Wagon R LXI CNG      Mumbai  2010              72000   \n",
       "1     Hyundai Creta 1.6 CRDi SX Option        Pune  2015              41000   \n",
       "2                         Honda Jazz V     Chennai  2011              46000   \n",
       "3                    Maruti Ertiga VDI     Chennai  2012              87000   \n",
       "4      Audi A4 New 2.0 TDI Multitronic  Coimbatore  2013              40670   \n",
       "...                                ...         ...   ...                ...   \n",
       "6013                  Maruti Swift VDI       Delhi  2014              27365   \n",
       "6014          Hyundai Xcent 1.1 CRDi S      Jaipur  2015             100000   \n",
       "6015             Mahindra Xylo D4 BSIV      Jaipur  2012              55000   \n",
       "6016                Maruti Wagon R VXI     Kolkata  2013              46000   \n",
       "6017             Chevrolet Beat Diesel   Hyderabad  2011              47000   \n",
       "\n",
       "     Fuel_Type Transmission Owner_Type  Mileage  Engine   Power  Seats  \\\n",
       "0          CNG       Manual      First    26.60   998.0   58.16    5.0   \n",
       "1       Diesel       Manual      First    19.67  1582.0  126.20    5.0   \n",
       "2       Petrol       Manual      First    18.20  1199.0   88.70    5.0   \n",
       "3       Diesel       Manual      First    20.77  1248.0   88.76    7.0   \n",
       "4       Diesel    Automatic     Second    15.20  1968.0  140.80    5.0   \n",
       "...        ...          ...        ...      ...     ...     ...    ...   \n",
       "6013    Diesel       Manual      First    28.40  1248.0   74.00    5.0   \n",
       "6014    Diesel       Manual      First    24.40  1120.0   71.00    5.0   \n",
       "6015    Diesel       Manual     Second    14.00  2498.0  112.00    8.0   \n",
       "6016    Petrol       Manual      First    18.90   998.0   67.10    5.0   \n",
       "6017    Diesel       Manual      First    25.44   936.0   57.60    5.0   \n",
       "\n",
       "      New_price  Price  kilometers_driven_log  price_log  CarAge      Brand  \\\n",
       "0         5.510   1.75              11.184421   0.559616    2010     MARUTI   \n",
       "1        16.060  12.50              10.621327   2.525729    2015    HYUNDAI   \n",
       "2         8.610   4.50              10.736397   1.504077    2011      HONDA   \n",
       "3        11.215   6.00              11.373663   1.791759    2012     MARUTI   \n",
       "4           NaN  17.74              10.613246   2.875822    2013       AUDI   \n",
       "...         ...    ...                    ...        ...     ...        ...   \n",
       "6013      7.880   4.75              10.217020   1.558145    2014     MARUTI   \n",
       "6014        NaN   4.00              11.512925   1.386294    2015    HYUNDAI   \n",
       "6015     11.690   2.90              10.915088   1.064711    2012   MAHINDRA   \n",
       "6016      5.510   2.65              10.736397   0.974560    2013     MARUTI   \n",
       "6017        NaN   2.50              10.757903   0.916291    2011  CHEVROLET   \n",
       "\n",
       "           Model  \n",
       "0         WagonR  \n",
       "1       Creta1.6  \n",
       "2          JazzV  \n",
       "3      ErtigaVDI  \n",
       "4          A4New  \n",
       "...          ...  \n",
       "6013    SwiftVDI  \n",
       "6014    Xcent1.1  \n",
       "6015      XyloD4  \n",
       "6016      WagonR  \n",
       "6017  BeatDiesel  \n",
       "\n",
       "[6018 rows x 18 columns]"
      ]
     },
     "execution_count": 311,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cars_data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0L-oAMItxLP-"
   },
   "source": [
    "### **Split the Data**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Aat-Ne-ZVL3e"
   },
   "source": [
    "- Step1: Seperating the independent variables (X) and the dependent variable (y). \n",
    "- Step2: Encode the categorical variables in X using pd.dummies.\n",
    "- Step3: Split the data into train and test using train_test_split."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Cwh_IhfqVL3f"
   },
   "source": [
    "### Columns to be dropped before splitting the data\n",
    "\n",
    "**Columns:**\n",
    "\n",
    "- Name\n",
    "- Price\n",
    "- price_log\n",
    "- Kilometers_Driven\n",
    "- New_price\n",
    "- Model\n",
    "- Year\n",
    "\n",
    "**Justification:**\n",
    "\n",
    "- Name: Contains unique car names- not useful for prediction. 'Brand' and 'Model' provide more structured data.\n",
    "- Price: Target variable- should not be in X to prevent data leakage.\n",
    "- price_log: Transformed target variable- should not be in X to prevent data leakage.\n",
    "- Kilometers_Driven: Redundant due to the use of log-transformed 'kilometers_driven_log'.\n",
    "- New_price: High missing value percentage (62.88%)- dropping it maintains data integrity.\n",
    "- Model: One-hot encoded into multiple binary columns- redundant.\n",
    "- Year: Information captured in 'CarAge'; redundant."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 312,
   "metadata": {
    "id": "NTly1jIxtOS8"
   },
   "outputs": [],
   "source": [
    "# Step-1: Separating the independent variables (X) and the dependent variable (y)\n",
    "X = cars_data.drop(['Name', 'Price', 'price_log', 'Kilometers_Driven', 'New_price', 'Model', 'Year'], axis=1)\n",
    "y = cars_data[[\"price_log\", \"Price\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 313,
   "metadata": {
    "id": "vzCLGMzbVL3f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mileage                      float64\n",
      "Engine                       float64\n",
      "Power                        float64\n",
      "Seats                        float64\n",
      "kilometers_driven_log        float64\n",
      "CarAge                         int64\n",
      "Location_Bangalore              bool\n",
      "Location_Chennai                bool\n",
      "Location_Coimbatore             bool\n",
      "Location_Delhi                  bool\n",
      "Location_Hyderabad              bool\n",
      "Location_Jaipur                 bool\n",
      "Location_Kochi                  bool\n",
      "Location_Kolkata                bool\n",
      "Location_Mumbai                 bool\n",
      "Location_Pune                   bool\n",
      "Fuel_Type_Diesel                bool\n",
      "Fuel_Type_Electric              bool\n",
      "Fuel_Type_LPG                   bool\n",
      "Fuel_Type_Petrol                bool\n",
      "Transmission_Manual             bool\n",
      "Owner_Type_Fourth & Above       bool\n",
      "Owner_Type_Second               bool\n",
      "Owner_Type_Third                bool\n",
      "Brand_AUDI                      bool\n",
      "Brand_BENTLEY                   bool\n",
      "Brand_BMW                       bool\n",
      "Brand_CHEVROLET                 bool\n",
      "Brand_DATSUN                    bool\n",
      "Brand_FIAT                      bool\n",
      "Brand_FORCE                     bool\n",
      "Brand_FORD                      bool\n",
      "Brand_HONDA                     bool\n",
      "Brand_HYUNDAI                   bool\n",
      "Brand_ISUZU                     bool\n",
      "Brand_JAGUAR                    bool\n",
      "Brand_JEEP                      bool\n",
      "Brand_LAMBORGHINI               bool\n",
      "Brand_LAND                      bool\n",
      "Brand_MAHINDRA                  bool\n",
      "Brand_MARUTI                    bool\n",
      "Brand_MERCEDES-BENZ             bool\n",
      "Brand_MINI                      bool\n",
      "Brand_MITSUBISHI                bool\n",
      "Brand_NISSAN                    bool\n",
      "Brand_PORSCHE                   bool\n",
      "Brand_RENAULT                   bool\n",
      "Brand_SKODA                     bool\n",
      "Brand_SMART                     bool\n",
      "Brand_TATA                      bool\n",
      "Brand_TOYOTA                    bool\n",
      "Brand_VOLKSWAGEN                bool\n",
      "Brand_VOLVO                     bool\n",
      "dtype: object\n",
      "price_log    float64\n",
      "Price        float64\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "# Step-2: Encode the categorical variables in X using pd.get_dummies\n",
    "X = pd.get_dummies(X, drop_first=True)\n",
    "\n",
    "# Print out the data types to verify\n",
    "print(X.dtypes)\n",
    "print(y.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 314,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mileage                      float64\n",
      "Engine                       float64\n",
      "Power                        float64\n",
      "Seats                        float64\n",
      "kilometers_driven_log        float64\n",
      "CarAge                         int64\n",
      "Location_Bangalore             int64\n",
      "Location_Chennai               int64\n",
      "Location_Coimbatore            int64\n",
      "Location_Delhi                 int64\n",
      "Location_Hyderabad             int64\n",
      "Location_Jaipur                int64\n",
      "Location_Kochi                 int64\n",
      "Location_Kolkata               int64\n",
      "Location_Mumbai                int64\n",
      "Location_Pune                  int64\n",
      "Fuel_Type_Diesel               int64\n",
      "Fuel_Type_Electric             int64\n",
      "Fuel_Type_LPG                  int64\n",
      "Fuel_Type_Petrol               int64\n",
      "Transmission_Manual            int64\n",
      "Owner_Type_Fourth & Above      int64\n",
      "Owner_Type_Second              int64\n",
      "Owner_Type_Third               int64\n",
      "Brand_AUDI                     int64\n",
      "Brand_BENTLEY                  int64\n",
      "Brand_BMW                      int64\n",
      "Brand_CHEVROLET                int64\n",
      "Brand_DATSUN                   int64\n",
      "Brand_FIAT                     int64\n",
      "Brand_FORCE                    int64\n",
      "Brand_FORD                     int64\n",
      "Brand_HONDA                    int64\n",
      "Brand_HYUNDAI                  int64\n",
      "Brand_ISUZU                    int64\n",
      "Brand_JAGUAR                   int64\n",
      "Brand_JEEP                     int64\n",
      "Brand_LAMBORGHINI              int64\n",
      "Brand_LAND                     int64\n",
      "Brand_MAHINDRA                 int64\n",
      "Brand_MARUTI                   int64\n",
      "Brand_MERCEDES-BENZ            int64\n",
      "Brand_MINI                     int64\n",
      "Brand_MITSUBISHI               int64\n",
      "Brand_NISSAN                   int64\n",
      "Brand_PORSCHE                  int64\n",
      "Brand_RENAULT                  int64\n",
      "Brand_SKODA                    int64\n",
      "Brand_SMART                    int64\n",
      "Brand_TATA                     int64\n",
      "Brand_TOYOTA                   int64\n",
      "Brand_VOLKSWAGEN               int64\n",
      "Brand_VOLVO                    int64\n",
      "dtype: object\n",
      "price_log    float64\n",
      "Price        float64\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "# Convert boolean columns to integers\n",
    "bool_cols = X.select_dtypes(include=['bool']).columns\n",
    "X[bool_cols] = X[bool_cols].astype(int)\n",
    "\n",
    "# Print out the data types to verify\n",
    "print(X.dtypes)\n",
    "print(y.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 315,
   "metadata": {
    "id": "JqVHLEHVRRKK"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train: (4212, 53)\n",
      "X_test: (1806, 53)\n",
      "y_train: (4212, 2)\n",
      "y_test: (1806, 2)\n"
     ]
    }
   ],
   "source": [
    "# Step-3: Split the data into train and test using train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)\n",
    "\n",
    "# Reset index of train and test sets\n",
    "X_train.reset_index(drop=True, inplace=True)\n",
    "X_test.reset_index(drop=True, inplace=True)\n",
    "y_train.reset_index(drop=True, inplace=True)\n",
    "y_test.reset_index(drop=True, inplace=True)\n",
    "\n",
    "print(\"X_train:\", X_train.shape)\n",
    "print(\"X_test:\", X_test.shape)\n",
    "print(\"y_train:\", y_train.shape)\n",
    "print(\"y_test:\", y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 316,
   "metadata": {
    "id": "glAH2vMFtOS8"
   },
   "outputs": [],
   "source": [
    "# Let us write a function for calculating r2_score and RMSE on train and test data\n",
    "# This function takes model as an input on which we have trained particular algorithm\n",
    "# The categorical column as the input and returns the boxplots and histograms for the variable\n",
    "\n",
    "def get_model_score(model, flag = True):\n",
    "    '''\n",
    "    model : regressor to predict values of X\n",
    "\n",
    "    '''\n",
    "    # Defining an empty list to store train and test results\n",
    "    score_list = [] \n",
    "    \n",
    "    pred_train = model.predict(X_train)\n",
    "    \n",
    "    pred_train_ = np.exp(pred_train)\n",
    "    \n",
    "    pred_test = model.predict(X_test)\n",
    "    \n",
    "    pred_test_ = np.exp(pred_test)\n",
    "    \n",
    "    train_r2 = metrics.r2_score(y_train['Price'], pred_train_)\n",
    "    \n",
    "    test_r2 = metrics.r2_score(y_test['Price'], pred_test_)\n",
    "    \n",
    "    train_rmse = metrics.mean_squared_error(y_train['Price'], pred_train_, squared = False)\n",
    "    \n",
    "    test_rmse = metrics.mean_squared_error(y_test['Price'], pred_test_, squared = False)\n",
    "    \n",
    "    # Adding all scores in the list\n",
    "    score_list.extend((train_r2, test_r2, train_rmse, test_rmse))\n",
    "    \n",
    "    # If the flag is set to True then only the following print statements will be dispayed, the default value is True\n",
    "    if flag == True: \n",
    "        \n",
    "        print(\"R-sqaure on training set : \", metrics.r2_score(y_train['Price'], pred_train_))\n",
    "        \n",
    "        print(\"R-square on test set : \", metrics.r2_score(y_test['Price'], pred_test_))\n",
    "        \n",
    "        print(\"RMSE on training set : \", np.sqrt(metrics.mean_squared_error(y_train['Price'], pred_train_)))\n",
    "        \n",
    "        print(\"RMSE on test set : \", np.sqrt(metrics.mean_squared_error(y_test['Price'], pred_test_)))\n",
    "    \n",
    "    # Returning the list with train and test scores\n",
    "    return score_list"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "o8qcI692VL3g"
   },
   "source": [
    "<hr>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gaj2riZFVL3g"
   },
   "source": [
    "For Regression Problems, some of the algorithms used are :<br>\n",
    "\n",
    "**1) Linear Regression** <br>\n",
    "**2) Ridge / Lasso Regression** <br>\n",
    "**3) Decision Trees** <br>\n",
    "**4) Random Forest** <br>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xwL33RaztOS9"
   },
   "source": [
    "### **Fitting a linear model**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kXj-84YCVL3h"
   },
   "source": [
    "Linear Regression can be implemented using: <br>\n",
    "\n",
    "**1) Sklearn:** https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html <br>\n",
    "**2) Statsmodels:** https://www.statsmodels.org/stable/regression.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {
    "id": "tABeKbbNVL3h"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-33 {color: black;background-color: white;}#sk-container-id-33 pre{padding: 0;}#sk-container-id-33 div.sk-toggleable {background-color: white;}#sk-container-id-33 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-33 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-33 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-33 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-33 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-33 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-33 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-33 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-33 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-33 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-33 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-33 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-33 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-33 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-33 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-33 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-33 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-33 div.sk-item {position: relative;z-index: 1;}#sk-container-id-33 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-33 div.sk-item::before, #sk-container-id-33 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-33 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-33 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-33 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-33 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-33 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-33 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-33 div.sk-label-container {text-align: center;}#sk-container-id-33 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-33 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-33\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-33\" type=\"checkbox\" checked><label for=\"sk-estimator-id-33\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 317,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a linear regression model using sklearn\n",
    "lr = LinearRegression()\n",
    "\n",
    "# Fit linear regression model\n",
    "lr.fit(X_train, y_train['price_log'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {
    "id": "ABshmMPAtOS9",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-sqaure on training set :  0.8606929828095062\n",
      "R-square on test set :  0.8671037715116187\n",
      "RMSE on training set :  4.1700164179450185\n",
      "RMSE on test set :  4.062896558719825\n"
     ]
    }
   ],
   "source": [
    "# Get score of the model\n",
    "LR_score = get_model_score(lr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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+PBhXXnllT5j7Nq1q8444wzvGHfu3KmPP/5Yl112mU9lZ8CAATrttNMCPk9qaqouv/xyrVq1SrW1tZKkpqYmPfLII5o8ebL69OkjSRo5cqT3tf/973+31R3xpZde0vvvv68ZM2YoJSVFklRSUiLDMPTQQw95j9u4caOampo0e/Zsy48drPr6et1444067rjjlJqaqtTUVPXo0UNffvml3n333bA/PwDEA4IUACSov/71r6qqqtLrr7+ujz/+WG+++aZGjx7tc0y3bt2UmZnpc9v+/fv1xRdfqEuXLkpLS/P52rdvnz799FNJ0v/93/9JknJzc31+PjU11RswOuJZaxVsMwVPd7uRI0e2GePKlSsDjrGj29rzox/9SF9//bUef/xxSc0hbu/evSopKfEeM3bsWK1Zs8Yb7o4++mgNGzZMjz32WMDH96xvmjJlir744gt98cUXysrK0umnn64nn3zS27Gws++ZHZdddpkeeOAB/eQnP9Gzzz6rV199VVVVVTryyCPV0NAQ9ucHgHjAGikASFBDhw71du3rSMsqjUffvn3Vp08fPfPMM+3+TM+ePSXJG5b27duno446ynv/4cOHvQGmI551Wh999JHf4zriWaPzxBNPaMCAAR0e13KMrbV3W3uOP/54ffe731VZWZmuvvpqlZWVKS8vT+PHj/c5bvLkyZo8ebIaGxtVWVmp0tJSXXbZZTrmmGNUXFzc7mPX1tbqySeflPRtVau1FStWaNasWT7vWf/+/S2NvSVP5a+xsdHndk/obDmmp59+Wr/+9a910003eW9vbGzUZ599Zvt5ASBRUZECAPiYOHGi/u///k9NTU0qKipq81VQUCCpuUOcJP3tb3/z+fm///3vOnz4sN/nGDJkiI499lg99NBDbS7sW0pPT5ekNlWQ8847T6mpqfrggw/aHaMnQBYUFKhfv3567LHHZJqm9+f37NmjV155xdobouapdv/+979VXl6udevW+UzDa2/MZ5xxhu68805JareLoMeKFSvU0NCg3/zmN3rhhRfafPXt29c7vW/8+PFKSUnR0qVL/Y41PT293aqRZ/PhN9980+f2tWvX+nxvGIZM0/S+9x7/+7//q6amJr/PDQDJhIoUAMDH9OnT9be//U0XXnihfvazn+m73/2u0tLS9NFHH+mFF17Q5MmTNWXKFA0dOlSXX3657rvvPqWlpWncuHHasWOHfve737WZLtieP/7xj5o0aZJGjRqluXPnKj8/X06nU88++6w3nA0fPlyS9Pvf/14zZsxQWlqaCgoKdMwxx+jWW2/VzTffrF27dun8889Xr169tH//fr366qvq3r27brnlFjkcDv3mN7/RT37yE02ZMkX/8z//oy+++EKLFi2yPLVPki699FLNmzdPl156qRobG3XVVVf53P+rX/1KH330kc455xwdffTR+uKLL/T73/9eaWlpOuOMMzp83AcffFC9evXS9ddf3+5asSuvvFKLFy/WG2+8oREjRujnP/+5fvOb36ihoUGXXnqpsrKy9M477+jTTz/VLbfc4n3PVq1apaVLl+qUU06Rw+FQUVGRcnNzNW7cOJWWlqpXr14aMGCANm3apFWrVvk8Z2ZmpsaOHau7775bffv21THHHKOXXnpJDz74oI444gjL7xkAJLxod7sAAISWp2tfVVWV3+NmzJhhdu/evd37XC6X+bvf/c4cMWKE2bVrV7NHjx5mYWGhefXVV5vvvfee97jGxkZz/vz5ZnZ2ttm1a1dz1KhRZkVFhTlgwICAXftM0zQrKirMCy64wMzKyjLT09PNY489tk0XwIULF5p5eXmmw+Fo8xhr1qwxzzrrLDMzM9NMT083BwwYYF5yySXm888/7/MY//u//2sOHjzY7NKlizlkyBDzoYceMmfMmBGwa19Ll112mSnJHD16dJv7nn76afOCCy4wjzrqKLNLly5mdna2eeGFF5qbN2/u8PHeeOMNU5I5Z86cDo+prq42Jfl0J/zrX/9qjhw50nteTjrpJJ+Oe5999pl5ySWXmEcccYRpGIbZ8p/6vXv3mpdcconZu3dvMysry7z88svNrVu3tuna99FHH5nf+973zF69epk9e/Y0zz//fHPHjh2WzysAJAPDNFvMdQAAAAAABMQaKQAAAACwiSAFAAAAADYRpAAAAADApqgGqZdfflmTJk1SXl6eDMPQmjVr2hzz7rvv6qKLLlJWVpZ69uypUaNGyel0eu9vbGzUddddp759+6p79+666KKLgt6XBAAAAACsiGqQ+vLLLzVixAg98MAD7d7/wQcf6PTTT1dhYaFefPFFvfHGG/rlL3/p0yJ2zpw5Wr16tR5//HGVl5ervr5eEydOZK8LAAAAAGETM137DMPQ6tWrdfHFF3tvmz59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99xzGjJkiN544w2NHz9e9913ny699FLt2rVLxx57rLZt2+azZmvy5Mk64ogj9PDDD7f7uI2NjWps/HYRb11dnfr370+QAgBEBCEgdFwNLjXWNSo9M53pkABCIqwb8p599tlatWqVjjjiCPXt29fnSS+++GL961//CuZh27jxxhtVW1urwsJCpaSkqKmpSbfffrsuvfRSSdK+ffskSTk5OT4/l5OT493bqj2lpaW65ZZbQjJGAADsyh+dr/zR+YSAEEjLSOO9AxAVQa2RevHFF3Xo0KE2t3/99dfavHlzpwflsXLlSj366KNasWKFtm3bpocffli/+93v2lSaDMO3C5Jpmm1ua2nhwoWqra31fnmmJwIAEElpGWnqkdODIAAAcchWRerNN9/0/v933nnHWxGSpKamJj3zzDM66qijQja4BQsW6KabbtL06dMlScOHD9eePXtUWlqqGTNmKDc3V1JzZapfv37enztw4ECbKlVL6enpSk+nrSwAAACA4NgKUieeeKIMw5BhGDr77LPb3J+RkeHTUa+zvvrqKzkcvkWzlJQUb/vzgQMHKjc3Vxs3bvSukTp06JBeeukl3XnnnSEbBwAAAAC0ZCtI7d69W6ZpatCgQXr11Vd15JFHeu/r0qWLsrOzlZKSErLBTZo0Sbfffrvy8/N1wgkn6PXXX9fixYv1ox/9SFLzlL45c+bojjvu0ODBgzV48GDdcccd6tatmy677LKQjQMAYA9rfwAAic5WkBowYIAk+WyIG07333+/fvnLX2rWrFk6cOCA8vLydPXVV+tXv/qV95gbbrhBDQ0NmjVrlj7//HOdeuqpeu6559SzZ8+IjBEA8C1nuVMViyu086md3m50BZMLVDy/mG50CYSgDABBtj8vLS1VTk6OtzLk8dBDD+mTTz7x7ukUL9hHCgA6r2pplTbM3sD+SAmMoAwgGVjNBkF17fvTn/6kwsLCNrefcMIJWrZsWTAPCQCIY85ypzbM3iCZ8glR0jffm9L6Wevl3OKM0gjRWVVLq1Q2tkw162pkups/gzXdpmrW1ahsTJm2Ltsa5RECQGQFFaRad8nzOPLII7V3795ODwoAEF8qFlfIkeL/nxRHikOV91ZGaEQIJYIyALQVVJDq37+/tmzZ0ub2LVu2KC8vr9ODAgDED1eDSzuf2tnmArs192G3qldXy9XgitDIECoE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/vvf/+ryyy9Xnz591K1bN5144ol67bXXvPdzXuPP4cOH9Ytf/EIDBw5URkaGBg0apFtvvVVut9t7DOc19r388suaNGmS8vLyZBiG1qxZ43O/lXPY2Nio6667Tn379lX37t110UUX6aOPPorgq0Br/s6ry+XSjTfeqOHDh6t79+7Ky8vTlVdeqY8//tjnMWL5vBKkomzlypWaM2eObr75Zr3++usaM2aMLrjgAjmdzmgPDRa99NJLmj17tiorK7Vx40YdPnxY48eP15dffuk95q677tLixYv1wAMPqKqqSrm5uTr33HN18ODBKI4cVlRVVenPf/6zvvOd7/jczjmNP59//rlGjx6ttLQ0/fOf/9Q777yje+65R0cccYT3GM5r/Lnzzju1bNkyPfDAA3r33Xd111136e6779b999/vPYbzGvu+/PJLjRgxQg888EC791s5h3PmzNHq1av1+OOPq7y8XPX19Zo4caKampoi9TLQir/z+tVXX2nbtm365S9/qW3btmnVqlWqqanRRRdd5HNcTJ9XE1H13e9+15w5c6bPbYWFheZNN90UpRGhsw4cOGBKMl966SXTNE3T7Xabubm55m9/+1vvMV9//bWZlZVlLlu2LFrDhAUHDx40Bw8ebG7cuNE844wzzJ/97GemaXJO49WNN95onn766R3ez3mNTxMmTDB/9KMf+dw2depU8/LLLzdNk/MajySZq1ev9n5v5Rx+8cUXZlpamvn44497j/nvf/9rOhwO85lnnonY2NGx1ue1Pa+++qopydyzZ49pmrF/XqlIRdGhQ4f02muvafz48T63jx8/Xq+88kqURoXOqq2tlST17t1bkrR7927t27fP5zynp6frjDPO4DzHuNmzZ2vChAkaN26cz+2c0/i0du1aFRUV6fvf/76ys7N10kkn6S9/+Yv3fs5rfDr99NO1adMm1dTUSJLeeOMNlZeX68ILL5TEeU0EVs7ha6+9JpfL5XNMXl6ehg0bxnmOI7W1tTIMwztTINbPa2q0B5DMPv30UzU1NSknJ8fn9pycHO3bty9Ko0JnmKapefPm6fTTT9ewYcMkyXsu2zvPe/bsifgYYc3jjz+ubdu2qaqqqs19nNP4tGvXLi1dulTz5s3Tz3/+c7366qv66U9/qvT0dF155ZWc1zh14403qra2VoWFhUpJSVFTU5Nuv/12XXrp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",
      "text/plain": [
       "<Figure size 1000x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting Predicted vs Actual\n",
    "plt.figure(figsize=(10,7))\n",
    "plt.scatter(y=np.exp(lr.predict(X_train)), x=np.exp(y_train['price_log'].values), s=50, c='purple')\n",
    "plt.xlabel(\"Actual\")\n",
    "plt.ylabel(\"Predicted\")\n",
    "plt.title(\"Predicted vs Actual\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NgNkZ0HctOS9"
   },
   "source": [
    "**Observations from results:**\n",
    "\n",
    "- **Model Fit:** The high R-squared values on both the training and test sets indicate that the model explains a large portion of the variance in the data, showing a strong fit.\n",
    "\n",
    "- **Generalization:** The slight increase in R-squared and decrease in RMSE on the test set compared to the training set suggests that the model generalizes well to new data and is not overfitting.\n",
    "\n",
    "- **Error Metrics:** The RMSE values provide an estimate of the average prediction error. Lower RMSE values on both scales confirm that the model predictions are close to the actual values."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RNN1nlPqIrdC"
   },
   "source": [
    "**Important variables of Linear Regression**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NyKxdVyhVL3i"
   },
   "source": [
    "Building a model using statsmodels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 320,
   "metadata": {
    "id": "vHyqhuVMIrdC",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:              price_log   R-squared:                       0.925\n",
      "Model:                            OLS   Adj. R-squared:                  0.924\n",
      "Method:                 Least Squares   F-statistic:                     961.5\n",
      "Date:                Sat, 08 Jun 2024   Prob (F-statistic):               0.00\n",
      "Time:                        19:15:40   Log-Likelihood:                 47.535\n",
      "No. Observations:                4212   AIC:                             12.93\n",
      "Df Residuals:                    4158   BIC:                             355.6\n",
      "Df Model:                          53                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "=============================================================================================\n",
      "                                coef    std err          t      P>|t|      [0.025      0.975]\n",
      "---------------------------------------------------------------------------------------------\n",
      "const                      -239.0507      3.379    -70.739      0.000    -245.676    -232.425\n",
      "Mileage                      -0.0168      0.002     -9.594      0.000      -0.020      -0.013\n",
      "Engine                        0.0001      2e-05      7.445      0.000       0.000       0.000\n",
      "Power                         0.0050      0.000     23.850      0.000       0.005       0.005\n",
      "Seats                         0.0380      0.007      5.226      0.000       0.024       0.052\n",
      "kilometers_driven_log        -0.0654      0.007     -9.378      0.000      -0.079      -0.052\n",
      "CarAge                        0.1197      0.002     71.834      0.000       0.116       0.123\n",
      "Location_Bangalore            0.1717      0.024      7.056      0.000       0.124       0.219\n",
      "Location_Chennai              0.0212      0.023      0.920      0.358      -0.024       0.067\n",
      "Location_Coimbatore           0.1030      0.022      4.656      0.000       0.060       0.146\n",
      "Location_Delhi               -0.0562      0.022     -2.498      0.013      -0.100      -0.012\n",
      "Location_Hyderabad            0.1107      0.022      5.135      0.000       0.068       0.153\n",
      "Location_Jaipur              -0.0667      0.024     -2.811      0.005      -0.113      -0.020\n",
      "Location_Kochi               -0.0290      0.022     -1.307      0.191      -0.072       0.015\n",
      "Location_Kolkata             -0.2383      0.023    -10.497      0.000      -0.283      -0.194\n",
      "Location_Mumbai              -0.0394      0.022     -1.828      0.068      -0.082       0.003\n",
      "Location_Pune                -0.0347      0.022     -1.561      0.119      -0.078       0.009\n",
      "Fuel_Type_Diesel              0.2050      0.040      5.093      0.000       0.126       0.284\n",
      "Fuel_Type_Electric            1.0549      0.245      4.304      0.000       0.574       1.535\n",
      "Fuel_Type_LPG                -0.1223      0.100     -1.220      0.223      -0.319       0.074\n",
      "Fuel_Type_Petrol             -0.0946      0.041     -2.282      0.023      -0.176      -0.013\n",
      "Transmission_Manual          -0.1094      0.012     -8.844      0.000      -0.134      -0.085\n",
      "Owner_Type_Fourth & Above     0.0050      0.109      0.047      0.963      -0.208       0.218\n",
      "Owner_Type_Second            -0.0613      0.011     -5.618      0.000      -0.083      -0.040\n",
      "Owner_Type_Third             -0.1295      0.028     -4.600      0.000      -0.185      -0.074\n",
      "Brand_AUDI                    0.4618      0.244      1.889      0.059      -0.017       0.941\n",
      "Brand_BENTLEY                 0.0749      0.349      0.214      0.830      -0.610       0.760\n",
      "Brand_BMW                     0.4187      0.245      1.712      0.087      -0.061       0.898\n",
      "Brand_CHEVROLET              -0.4535      0.244     -1.856      0.064      -0.933       0.026\n",
      "Brand_DATSUN                 -0.6194      0.257     -2.414      0.016      -1.123      -0.116\n",
      "Brand_FIAT                   -0.4293      0.249     -1.723      0.085      -0.918       0.059\n",
      "Brand_FORCE                   0.0041      0.297      0.014      0.989      -0.578       0.587\n",
      "Brand_FORD                   -0.2149      0.243     -0.882      0.378      -0.692       0.263\n",
      "Brand_HONDA                  -0.0777      0.244     -0.319      0.750      -0.555       0.400\n",
      "Brand_HYUNDAI                -0.1388      0.243     -0.570      0.568      -0.616       0.338\n",
      "Brand_ISUZU                  -0.3643      0.280     -1.299      0.194      -0.914       0.185\n",
      "Brand_JAGUAR                  0.4386      0.248      1.765      0.078      -0.048       0.926\n",
      "Brand_JEEP                    0.0436      0.254      0.171      0.864      -0.455       0.542\n",
      "Brand_LAMBORGHINI             0.4695      0.349      1.347      0.178      -0.214       1.153\n",
      "Brand_LAND                    0.7368      0.247      2.986      0.003       0.253       1.221\n",
      "Brand_MAHINDRA               -0.2805      0.244     -1.150      0.250      -0.759       0.198\n",
      "Brand_MARUTI                 -0.1429      0.243     -0.587      0.557      -0.620       0.334\n",
      "Brand_MERCEDES-BENZ           0.4853      0.244      1.987      0.047       0.006       0.964\n",
      "Brand_MINI                    0.9119      0.251      3.639      0.000       0.421       1.403\n",
      "Brand_MITSUBISHI              0.1379      0.249      0.553      0.581      -0.351       0.627\n",
      "Brand_NISSAN                 -0.1753      0.245     -0.716      0.474      -0.655       0.305\n",
      "Brand_PORSCHE                 0.2030      0.254      0.798      0.425      -0.296       0.702\n",
      "Brand_RENAULT                -0.2076      0.244     -0.849      0.396      -0.687       0.272\n",
      "Brand_SKODA                  -0.0609      0.244     -0.249      0.803      -0.540       0.418\n",
      "Brand_SMART                   0.1111      0.343      0.324      0.746      -0.562       0.784\n",
      "Brand_TATA                   -0.6682      0.244     -2.739      0.006      -1.146      -0.190\n",
      "Brand_TOYOTA                  0.0774      0.244      0.318      0.751      -0.400       0.555\n",
      "Brand_VOLKSWAGEN             -0.1620      0.244     -0.665      0.506      -0.640       0.316\n",
      "Brand_VOLVO                   0.2559      0.251      1.019      0.308      -0.236       0.748\n",
      "==============================================================================\n",
      "Omnibus:                     1303.840   Durbin-Watson:                   2.010\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):            30059.589\n",
      "Skew:                          -0.933   Prob(JB):                         0.00\n",
      "Kurtosis:                      15.954   Cond. No.                     2.38e+06\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 2.38e+06. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    }
   ],
   "source": [
    "# Fitting a linear model using statsmodels\n",
    "import statsmodels.api as sm\n",
    "\n",
    "# Statsmodel api does not add a constant by default. We need to add it explicitly\n",
    "X_train_sm = sm.add_constant(X_train)\n",
    "X_test_sm = sm.add_constant(X_test)\n",
    "\n",
    "def build_ols_model(train):\n",
    "    olsmodel = sm.OLS(y_train[\"price_log\"], train)\n",
    "    return olsmodel.fit()\n",
    "\n",
    "olsmodel1 = build_ols_model(X_train_sm)\n",
    "print(olsmodel1.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "metadata": {
    "id": "C01l5EucVL3i"
   },
   "outputs": [],
   "source": [
    "# Retrive Coeff values, p-values and store them in the dataframe\n",
    "olsmod = pd.DataFrame(olsmodel1.params, columns = ['coef'])\n",
    "\n",
    "olsmod['pval'] = olsmodel1.pvalues"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {
    "id": "FbQQZacV9WMm"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>coef</th>\n",
       "      <th>pval</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Brand_MERCEDES-BENZ</th>\n",
       "      <td>0.485284</td>\n",
       "      <td>4.697351e-02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fuel_Type_Petrol</th>\n",
       "      <td>-0.094561</td>\n",
       "      <td>2.253076e-02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Brand_DATSUN</th>\n",
       "      <td>-0.619446</td>\n",
       "      <td>1.582547e-02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Location_Delhi</th>\n",
       "      <td>-0.056184</td>\n",
       "      <td>1.252295e-02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Brand_TATA</th>\n",
       "      <td>-0.668168</td>\n",
       "      <td>6.187912e-03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Location_Jaipur</th>\n",
       "      <td>-0.066666</td>\n",
       "      <td>4.954559e-03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Brand_LAND</th>\n",
       "      <td>0.736796</td>\n",
       "      <td>2.843420e-03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Brand_MINI</th>\n",
       "      <td>0.911944</td>\n",
       "      <td>2.764931e-04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fuel_Type_Electric</th>\n",
       "      <td>1.054933</td>\n",
       "      <td>1.717452e-05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Owner_Type_Third</th>\n",
       "      <td>-0.129455</td>\n",
       "      <td>4.342547e-06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Location_Coimbatore</th>\n",
       "      <td>0.103023</td>\n",
       "      <td>3.325267e-06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fuel_Type_Diesel</th>\n",
       "      <td>0.204965</td>\n",
       "      <td>3.687169e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Location_Hyderabad</th>\n",
       "      <td>0.110698</td>\n",
       "      <td>2.952985e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Seats</th>\n",
       "      <td>0.038042</td>\n",
       "      <td>1.819809e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Owner_Type_Second</th>\n",
       "      <td>-0.061259</td>\n",
       "      <td>2.058341e-08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Location_Bangalore</th>\n",
       "      <td>0.171697</td>\n",
       "      <td>1.996510e-12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Engine</th>\n",
       "      <td>0.000149</td>\n",
       "      <td>1.172032e-13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Transmission_Manual</th>\n",
       "      <td>-0.109378</td>\n",
       "      <td>1.338846e-18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>kilometers_driven_log</th>\n",
       "      <td>-0.065387</td>\n",
       "      <td>1.071847e-20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mileage</th>\n",
       "      <td>-0.016756</td>\n",
       "      <td>1.413432e-21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Location_Kolkata</th>\n",
       "      <td>-0.238277</td>\n",
       "      <td>1.852274e-25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Power</th>\n",
       "      <td>0.004991</td>\n",
       "      <td>6.100277e-118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CarAge</th>\n",
       "      <td>0.119684</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>const</th>\n",
       "      <td>-239.050685</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                             coef           pval\n",
       "Brand_MERCEDES-BENZ      0.485284   4.697351e-02\n",
       "Fuel_Type_Petrol        -0.094561   2.253076e-02\n",
       "Brand_DATSUN            -0.619446   1.582547e-02\n",
       "Location_Delhi          -0.056184   1.252295e-02\n",
       "Brand_TATA              -0.668168   6.187912e-03\n",
       "Location_Jaipur         -0.066666   4.954559e-03\n",
       "Brand_LAND               0.736796   2.843420e-03\n",
       "Brand_MINI               0.911944   2.764931e-04\n",
       "Fuel_Type_Electric       1.054933   1.717452e-05\n",
       "Owner_Type_Third        -0.129455   4.342547e-06\n",
       "Location_Coimbatore      0.103023   3.325267e-06\n",
       "Fuel_Type_Diesel         0.204965   3.687169e-07\n",
       "Location_Hyderabad       0.110698   2.952985e-07\n",
       "Seats                    0.038042   1.819809e-07\n",
       "Owner_Type_Second       -0.061259   2.058341e-08\n",
       "Location_Bangalore       0.171697   1.996510e-12\n",
       "Engine                   0.000149   1.172032e-13\n",
       "Transmission_Manual     -0.109378   1.338846e-18\n",
       "kilometers_driven_log   -0.065387   1.071847e-20\n",
       "Mileage                 -0.016756   1.413432e-21\n",
       "Location_Kolkata        -0.238277   1.852274e-25\n",
       "Power                    0.004991  6.100277e-118\n",
       "CarAge                   0.119684   0.000000e+00\n",
       "const                 -239.050685   0.000000e+00"
      ]
     },
     "execution_count": 322,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Filter by significant p-value (pval <= 0.05) and sort descending by Odds ratio\n",
    "\n",
    "olsmod = olsmod.sort_values(by = \"pval\", ascending = False)\n",
    "\n",
    "pval_filter = olsmod['pval']<= 0.05\n",
    "\n",
    "olsmod[pval_filter]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {
    "id": "yjplRhssIrdC"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1mMost overall significant categorical varaibles of LINEAR REGRESSION  are \u001b[95m :\n",
      " ['Brand_AUDI', 'Brand_BENTLEY', 'Brand_BMW', 'Brand_CHEVROLET', 'Brand_DATSUN', 'Brand_FIAT', 'Brand_FORCE', 'Brand_FORD', 'Brand_HONDA', 'Brand_HYUNDAI', 'Brand_ISUZU', 'Brand_JAGUAR', 'Brand_JEEP', 'Brand_LAMBORGHINI', 'Brand_LAND', 'Brand_MAHINDRA', 'Brand_MARUTI', 'Brand_MERCEDES-BENZ', 'Brand_MINI', 'Brand_MITSUBISHI', 'Brand_NISSAN', 'Brand_PORSCHE', 'Brand_RENAULT', 'Brand_SKODA', 'Brand_SMART', 'Brand_TATA', 'Brand_TOYOTA', 'Brand_VOLKSWAGEN', 'Brand_VOLVO', 'Fuel_Type_Diesel', 'Fuel_Type_Electric', 'Fuel_Type_LPG', 'Fuel_Type_Petrol', 'Location_Bangalore', 'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi', 'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi', 'Location_Kolkata', 'Location_Mumbai', 'Location_Pune', 'Owner_Type_Fourth & Above', 'Owner_Type_Second', 'Owner_Type_Third', 'Seats', 'Engine', 'Transmission_Manual', 'kilometers_driven_log', 'Mileage', 'Power', 'CarAge']\n"
     ]
    }
   ],
   "source": [
    "# We are looking at overall significant varaible\n",
    "\n",
    "pval_filter = olsmod['pval']<= 0.05\n",
    "imp_vars = olsmod[pval_filter].index.tolist()\n",
    "\n",
    "# We are going to get overall varaibles (un-one-hot encoded varables) from categorical varaibles\n",
    "sig_var = []\n",
    "for col in imp_vars:\n",
    "    if '' in col:\n",
    "        first_part = col.split('_')[0]\n",
    "        for c in X.columns:\n",
    "            if first_part in c and c not in sig_var :\n",
    "                sig_var.append(c)\n",
    "\n",
    "                \n",
    "start = '\\033[1m'\n",
    "end = '\\033[95m'\n",
    "print(start+ 'Most overall significant categorical varaibles of LINEAR REGRESSION  are ' +end,':\\n', sig_var)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Key Insights from OLS Regression Results\n",
    "\n",
    "**Model Performance:**\n",
    "\n",
    "- **R-squared:** 92.5%, indicating strong explanatory power.\n",
    "- **Adjusted R-squared:** 92.4%, accounting for the number of predictors.\n",
    "\n",
    "**Model Significance:**\n",
    "\n",
    "- **F-statistic:** 961.5 (p-value = 0.00), confirming the model's overall significance.\n",
    "\n",
    "**Significant Predictors:**\n",
    "\n",
    "- **Positive Impact:**\n",
    "\n",
    "    - CarAge: Newer cars have a higher price (coef: 0.1197).\n",
    "    - Power: More powerful cars have a higher price (coef: 0.0050).\n",
    "    - Engine Size: Larger engines increase the price slightly (coef: 0.0001).\n",
    "    - Seats: More seats increase the price (coef: 0.0380).\n",
    "    - Fuel Types: Diesel (coef: 0.2050) and Electric (coef: 1.0549).\n",
    "    - Locations: Bangalore, Coimbatore, Hyderabad.\n",
    "\n",
    "- **Negative Impact:**\n",
    "\n",
    "    - Mileage: Higher mileage lowers the price (coef: -0.0168).\n",
    "    - Kilometers Driven (log): Higher values lower the price (coef: -0.0654).\n",
    "    - Fuel Type: Petrol (coef: -0.0946).\n",
    "    - Locations: Delhi, Jaipur, Kolkata.\n",
    "    - Ownership: Second (coef: -0.0613) and third (coef: -0.1295) owners.\n",
    "    - Transmission: Manual (coef: -0.1094).\n",
    "\n",
    "**Brand Influence:**\n",
    "\n",
    "- **Positive:**\n",
    "    - Mercedes-Benz (0.4853), MINI (0.9119), Land Rover (0.7368)\n",
    "\n",
    "- **Negative:**\n",
    "    - Datsun (-0.6194), TATA (-0.6682)\n",
    "\n",
    "**Multicollinearity Concerns:**\n",
    "\n",
    "- **Condition Number:** 2.38e+06, indicating potential multicollinearity.\n",
    "\n",
    "### Significant Variables (p-value <= 0.05):\n",
    "\n",
    "- **CarAge**\n",
    "- **Power**\n",
    "- **Mileage**\n",
    "- **Engine**\n",
    "- **Seats**\n",
    "- **Kilometers Driven (log)**\n",
    "- **Locations:** Bangalore, Coimbatore, Delhi, Hyderabad, Jaipur, Kolkata\n",
    "- **Fuel Types:** Diesel, Electric, Petrol\n",
    "- **Brands:** Mercedes-Benz, MINI, Land Rover, Datsun, TATA\n",
    "- **Transmission:** Manual\n",
    "- **Ownership:** Second and Third owners\n",
    "\n",
    "### Overall Significant Categorical Variables:\n",
    "\n",
    "- **Brands:** AUDI, BENTLEY, BMW, CHEVROLET, DATSUN, FIAT, FORCE, FORD, HONDA, HYUNDAI, ISUZU, JAGUAR, JEEP, LAMBORGHINI, LAND ROVER, MAHINDRA, MARUTI, MERCEDES-BENZ, MINI, MITSUBISHI, NISSAN, PORSCHE, RENAULT, SKODA, SMART, TATA, TOYOTA, VOLKSWAGEN, VOLVO\n",
    "- **Fuel Types:** Diesel, Electric, LPG, Petrol\n",
    "- **Locations:** Bangalore, Chennai, Coimbatore, Delhi, Hyderabad, Jaipur, Kochi, Kolkata, Mumbai, Pune\n",
    "- **Ownership Types:** Fourth & Above, Second, Third\n",
    "- **Other Variables:** Seats, Engine, Transmission (Manual), kilometers_driven_log, Mileage, Power, CarAge"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5uubmKLlVL3j"
   },
   "source": [
    "**Build Ridge / Lasso Regression similar to Linear Regression:**<br>\n",
    "\n",
    "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 324,
   "metadata": {
    "id": "9--bgDNhVL3j"
   },
   "outputs": [],
   "source": [
    "# Import Ridge/ Lasso Regression from sklearn\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.linear_model import Lasso"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 325,
   "metadata": {
    "id": "TwSuk8Q9VL3j"
   },
   "outputs": [],
   "source": [
    "# Create a Ridge regression model\n",
    "ridge = Ridge(alpha=1.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 326,
   "metadata": {
    "id": "5cWh1QK0VL3j"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-34 {color: black;background-color: white;}#sk-container-id-34 pre{padding: 0;}#sk-container-id-34 div.sk-toggleable {background-color: white;}#sk-container-id-34 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-34 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-34 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-34 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-34 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-34 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-34 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-34 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-34 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-34 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-34 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-34 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-34 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-34 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-34 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-34 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-34 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-34 div.sk-item {position: relative;z-index: 1;}#sk-container-id-34 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-34 div.sk-item::before, #sk-container-id-34 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-34 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-34 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-34 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-34 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-34 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-34 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-34 div.sk-label-container {text-align: center;}#sk-container-id-34 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-34 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-34\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Ridge()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-34\" type=\"checkbox\" checked><label for=\"sk-estimator-id-34\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">Ridge</label><div class=\"sk-toggleable__content\"><pre>Ridge()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "Ridge()"
      ]
     },
     "execution_count": 326,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fit Ridge regression model\n",
    "ridge.fit(X_train, y_train['price_log']) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 327,
   "metadata": {
    "id": "c2tIACYOVL3j"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-sqaure on training set :  0.8573236559746695\n",
      "R-square on test set :  0.8652405189006817\n",
      "RMSE on training set :  4.2201438458170575\n",
      "RMSE on test set :  4.091279052587326\n"
     ]
    }
   ],
   "source": [
    "# Get score of the model\n",
    "ridge_score = get_model_score(ridge)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Observations from Ridge Regression Results:\n",
    "\n",
    "**Model Fit:**\n",
    "\n",
    "- R-squared: **Training:** 0.8573, **Test:** 0.8652\n",
    "- RMSE: **Training:** 4.2201, **Test:** 4.0913\n",
    "\n",
    "**Generalization:**\n",
    "\n",
    "- Slightly higher R-squared and lower RMSE on the test set indicate good generalization with minimal overfitting.\n",
    "\n",
    "**Prediction Accuracy:**\n",
    "\n",
    "- Low RMSE values on both scales confirm accurate predictions close to actual values.\n",
    "\n",
    "**Conclusion:**\n",
    "\n",
    "- The Ridge regression model shows strong fit, excellent generalization, and high prediction accuracy."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "owyg5IpstOS9"
   },
   "source": [
    "### **Decision Tree** \n",
    "\n",
    "https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 328,
   "metadata": {
    "id": "GjZeRER6VL3k"
   },
   "outputs": [],
   "source": [
    "# Import Decision tree for Regression from sklearn\n",
    "from sklearn.tree import DecisionTreeRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 329,
   "metadata": {
    "id": "shBet9WztOS-"
   },
   "outputs": [],
   "source": [
    "# Create a decision tree regression model, use random_state = 1\n",
    "dtree = DecisionTreeRegressor(random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 330,
   "metadata": {
    "id": "eJxVYVXXtOS-"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-35 {color: black;background-color: white;}#sk-container-id-35 pre{padding: 0;}#sk-container-id-35 div.sk-toggleable {background-color: white;}#sk-container-id-35 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-35 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-35 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-35 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-35 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-35 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-35 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-35 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-35 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-35 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-35 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-35 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-35 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-35 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-35 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-35 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-35 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-35 div.sk-item {position: relative;z-index: 1;}#sk-container-id-35 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-35 div.sk-item::before, #sk-container-id-35 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-35 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-35 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-35 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-35 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-35 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-35 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-35 div.sk-label-container {text-align: center;}#sk-container-id-35 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-35 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-35\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeRegressor(random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-35\" type=\"checkbox\" checked><label for=\"sk-estimator-id-35\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeRegressor</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeRegressor(random_state=1)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeRegressor(random_state=1)"
      ]
     },
     "execution_count": 330,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fit decision tree regression model\n",
    "dtree.fit(X_train, y_train['price_log']) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 331,
   "metadata": {
    "id": "vGbEjda0tOS-"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-sqaure on training set :  0.9999965696959587\n",
      "R-square on test set :  0.7983636206173096\n",
      "RMSE on training set :  0.020692719736775493\n",
      "RMSE on test set :  5.00453674766219\n"
     ]
    }
   ],
   "source": [
    "# Get score of the model\n",
    "Dtree_model = get_model_score(dtree)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "UrCgLVKwtOS-"
   },
   "source": [
    "### Observations from Decision Tree Regression Results:\n",
    "\n",
    "**Model Fit:**\n",
    "\n",
    "- R-squared: **Training:** 0.99999, **Test:** 0.7984\n",
    "- RMSE: **Training:** 0.0207, **Test:** 5.0045\n",
    "\n",
    "**Generalization:**\n",
    "\n",
    "- The significant drop in R-squared and increase in RMSE on the test set compared to the training set indicates overfitting and poor generalization.\n",
    "\n",
    "**Prediction Accuracy:**\n",
    "\n",
    "- Extremely low RMSE on the training set but much higher RMSE on the test set suggests that the model does not predict well on unseen data.\n",
    "\n",
    "**Conclusion:**\n",
    "\n",
    "- The Decision Tree model overfits the training data and shows poor generalization to the test data. Consider tuning or using ensemble methods like Random Forest to improve performance."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6z2V0IwtVL3k"
   },
   "source": [
    "Print the importance of features in the tree building. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 332,
   "metadata": {
    "id": "E8ro_i9vIrdF"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                    Imp\n",
      "Power                      6.130113e-01\n",
      "CarAge                     2.356855e-01\n",
      "Engine                     5.099671e-02\n",
      "kilometers_driven_log      1.805011e-02\n",
      "Mileage                    1.419337e-02\n",
      "Brand_TATA                 5.353406e-03\n",
      "Brand_HONDA                4.704945e-03\n",
      "Location_Kolkata           4.666472e-03\n",
      "Transmission_Manual        4.604547e-03\n",
      "Brand_AUDI                 3.458510e-03\n",
      "Brand_MAHINDRA             3.405413e-03\n",
      "Seats                      3.188453e-03\n",
      "Brand_MINI                 3.169415e-03\n",
      "Location_Hyderabad         3.029701e-03\n",
      "Brand_SKODA                2.914090e-03\n",
      "Location_Coimbatore        2.504641e-03\n",
      "Brand_HYUNDAI              2.124056e-03\n",
      "Brand_VOLKSWAGEN           2.039523e-03\n",
      "Brand_TOYOTA               1.953610e-03\n",
      "Location_Delhi             1.937839e-03\n",
      "Location_Mumbai            1.717458e-03\n",
      "Owner_Type_Second          1.621995e-03\n",
      "Fuel_Type_Petrol           1.514219e-03\n",
      "Owner_Type_Third           1.400051e-03\n",
      "Brand_MARUTI               1.339037e-03\n",
      "Location_Bangalore         1.285552e-03\n",
      "Location_Chennai           1.108212e-03\n",
      "Brand_CHEVROLET            1.094941e-03\n",
      "Location_Jaipur            9.755268e-04\n",
      "Brand_LAND                 9.213142e-04\n",
      "Location_Kochi             9.213103e-04\n",
      "Brand_MERCEDES-BENZ        8.963489e-04\n",
      "Brand_FORD                 8.331745e-04\n",
      "Location_Pune              6.042638e-04\n",
      "Brand_PORSCHE              5.924083e-04\n",
      "Fuel_Type_Diesel           5.755306e-04\n",
      "Fuel_Type_Electric         4.135426e-04\n",
      "Brand_JEEP                 2.923842e-04\n",
      "Brand_FIAT                 2.208597e-04\n",
      "Brand_BMW                  1.978528e-04\n",
      "Brand_JAGUAR               1.256072e-04\n",
      "Brand_NISSAN               8.773187e-05\n",
      "Brand_RENAULT              8.699409e-05\n",
      "Brand_VOLVO                7.457037e-05\n",
      "Brand_ISUZU                5.580822e-05\n",
      "Brand_MITSUBISHI           2.901918e-05\n",
      "Owner_Type_Fourth & Above  2.258964e-05\n",
      "Fuel_Type_LPG              4.732144e-08\n",
      "Brand_LAMBORGHINI          0.000000e+00\n",
      "Brand_FORCE                0.000000e+00\n",
      "Brand_DATSUN               0.000000e+00\n",
      "Brand_SMART                0.000000e+00\n",
      "Brand_BENTLEY              0.000000e+00\n"
     ]
    }
   ],
   "source": [
    "print(pd.DataFrame(dtree.feature_importances_, columns = [\"Imp\"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "m9P0pzHHIrdG"
   },
   "source": [
    "### Observations and Insights from Decision Tree Feature Importances\n",
    "\n",
    "**Top Features:**\n",
    "\n",
    "- Power (61.30%): The most crucial factor influencing car prices.\n",
    "- CarAge (23.57%): Significant impact, indicating the effect of the car's age on its price.\n",
    "- Engine (5.10%): Important, but less influential than Power and CarAge.\n",
    "\n",
    "**Moderate Importance:**\n",
    "\n",
    "- Kilometers Driven (log) (1.81%): Moderately affects car prices.\n",
    "- Mileage (1.42%): Slight influence on the car's price.\n",
    "\n",
    "**Brand and Location Influence:**\n",
    "\n",
    "- Brand TATA (0.54%) and Brand HONDA (0.47%): Notable impact among brands.\n",
    "- Location Kolkata (0.47%) and Location Hyderabad (0.30%): Locations with noticeable effects on prices.\n",
    "\n",
    "**Transmission and Ownership:**\n",
    "\n",
    "- Transmission Manual (0.46%): Has a slight effect on car prices.\n",
    "- Owner Type Second (0.16%): Previous ownership slightly impacts the price.\n",
    "\n",
    "**Other Notable Features:**\n",
    "\n",
    "- Seats (0.32%): Minor influence on price.\n",
    "- Fuel Type Petrol (0.15%): Small effect among fuel types.\n",
    "\n",
    "**Least Important Features:**\n",
    "\n",
    "- Some brands and locations have minimal or no impact (e.g., Brand Lamborghini, Brand Force, Brand Datsun, Brand SMART, Brand Bentley).\n",
    "\n",
    "### Summary:\n",
    "\n",
    "- **Power** and **CarAge** are the most critical determinants of car price, reflecting the significance of a car's performance and age.\n",
    "- **Engine size** and **mileage-related** features moderately influence the price.\n",
    "- **Brand** and **location** play varying roles, with some brands and locations significantly impacting prices.\n",
    "- **Transmission type** and **ownership history** also contribute but to a lesser extent.\n",
    "- **Fuel types other than petrol** have minimal impact on the model's predictions.\n",
    "- **Several features have negligible influence**, suggesting they can potentially be excluded in future models for simplicity."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "q8eFynaNtOS-"
   },
   "source": [
    "### **Random Forest**\n",
    "\n",
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "metadata": {
    "id": "Jhw-0FsNVL3l"
   },
   "outputs": [],
   "source": [
    "# Import Randomforest for Regression from sklearn\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.datasets import make_regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "metadata": {
    "id": "-4S4FoDXtOS-"
   },
   "outputs": [],
   "source": [
    "# Create a Randomforest regression model\n",
    "rfr = RandomForestRegressor(n_estimators=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 335,
   "metadata": {
    "id": "gBlavhMTtOS-"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-36 {color: black;background-color: white;}#sk-container-id-36 pre{padding: 0;}#sk-container-id-36 div.sk-toggleable {background-color: white;}#sk-container-id-36 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-36 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-36 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-36 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-36 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-36 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-36 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-36 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-36 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-36 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-36 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-36 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-36 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-36 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-36 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-36 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-36 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-36 div.sk-item {position: relative;z-index: 1;}#sk-container-id-36 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-36 div.sk-item::before, #sk-container-id-36 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-36 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-36 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-36 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-36 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-36 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-36 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-36 div.sk-label-container {text-align: center;}#sk-container-id-36 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-36 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-36\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-36\" type=\"checkbox\" checked><label for=\"sk-estimator-id-36\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestRegressor()"
      ]
     },
     "execution_count": 335,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fit Randomforest regression model\n",
    "rfr.fit(X_train, y_train['price_log'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 336,
   "metadata": {
    "id": "VLDDeeAGtOS_"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-sqaure on training set :  0.9752326573189154\n",
      "R-square on test set :  0.8745798373053588\n",
      "RMSE on training set :  1.758291897262418\n",
      "RMSE on test set :  3.946963563228673\n"
     ]
    }
   ],
   "source": [
    "# Get score of the model\n",
    "RFR_model = get_model_score(rfr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZGNTRfaitOS_"
   },
   "source": [
    "### Observations from Random Forest Regression Results:\n",
    "\n",
    "**Model Fit:**\n",
    "\n",
    "- R-squared: **Training:** 0.9756, **Test:** 0.8690\n",
    "- RMSE: **Training:** 1.7465, **Test:** 4.0336\n",
    "\n",
    "**Generalization:**\n",
    "\n",
    "- High R-squared and low RMSE on both training and test sets indicate strong predictive performance and good generalization.\n",
    "- Small differences between training and test metrics suggest minimal overfitting.\n",
    "\n",
    "**Prediction Accuracy:**\n",
    "\n",
    "- Low RMSE values on both scales confirm accurate predictions close to actual values.\n",
    "\n",
    "**Conclusion:**\n",
    "\n",
    "- The Random Forest model shows robust performance, high accuracy, and excellent generalization capabilities."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "pgwyNxUuIrdG"
   },
   "source": [
    "**Feature Importance**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 337,
   "metadata": {
    "id": "AWRS7zISIrdG"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                Imp\n",
      "Power                      0.623611\n",
      "CarAge                     0.234310\n",
      "Engine                     0.036472\n",
      "kilometers_driven_log      0.019381\n",
      "Mileage                    0.015949\n",
      "Location_Kolkata           0.005341\n",
      "Transmission_Manual        0.005104\n",
      "Brand_TATA                 0.004713\n",
      "Seats                      0.004386\n",
      "Brand_HONDA                0.003679\n",
      "Location_Hyderabad         0.003284\n",
      "Brand_MAHINDRA             0.002999\n",
      "Brand_MERCEDES-BENZ        0.002776\n",
      "Brand_MINI                 0.002664\n",
      "Location_Coimbatore        0.002499\n",
      "Brand_AUDI                 0.002333\n",
      "Brand_SKODA                0.002174\n",
      "Brand_LAND                 0.001797\n",
      "Owner_Type_Second          0.001756\n",
      "Fuel_Type_Diesel           0.001718\n",
      "Brand_VOLKSWAGEN           0.001652\n",
      "Brand_TOYOTA               0.001608\n",
      "Location_Delhi             0.001559\n",
      "Location_Bangalore         0.001547\n",
      "Location_Mumbai            0.001524\n",
      "Brand_CHEVROLET            0.001393\n",
      "Brand_HYUNDAI              0.001388\n",
      "Fuel_Type_Petrol           0.001339\n",
      "Location_Pune              0.001332\n",
      "Location_Jaipur            0.001302\n",
      "Brand_MARUTI               0.001126\n",
      "Brand_PORSCHE              0.001125\n",
      "Location_Kochi             0.001096\n",
      "Location_Chennai           0.001068\n",
      "Owner_Type_Third           0.000970\n",
      "Brand_BMW                  0.000758\n",
      "Brand_FORD                 0.000520\n",
      "Fuel_Type_Electric         0.000454\n",
      "Brand_RENAULT              0.000236\n",
      "Brand_FIAT                 0.000223\n",
      "Brand_NISSAN               0.000187\n",
      "Brand_JAGUAR               0.000170\n",
      "Brand_MITSUBISHI           0.000152\n",
      "Brand_JEEP                 0.000085\n",
      "Brand_VOLVO                0.000083\n",
      "Brand_LAMBORGHINI          0.000055\n",
      "Owner_Type_Fourth & Above  0.000051\n",
      "Brand_DATSUN               0.000018\n",
      "Fuel_Type_LPG              0.000012\n",
      "Brand_ISUZU                0.000009\n",
      "Brand_BENTLEY              0.000005\n",
      "Brand_FORCE                0.000003\n",
      "Brand_SMART                0.000003\n"
     ]
    }
   ],
   "source": [
    "# Print important features similar to decision trees\n",
    "print(pd.DataFrame(rfr.feature_importances_, columns = [\"Imp\"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cG9ZD9ozIrdH"
   },
   "source": [
    "### Observations and Insights from Random Forest Feature Importances\n",
    "\n",
    "**Top Features:**\n",
    "\n",
    "- Power (62.01%): The most influential factor in predicting car prices.\n",
    "- CarAge (23.46%): Second most important factor, reflecting the impact of the car's age.\n",
    "- Engine (3.83%): Significant, though less than Power and CarAge.\n",
    "\n",
    "**Moderate Importance:**\n",
    "\n",
    "- Kilometers Driven (log) (1.94%): Moderately affects prices.\n",
    "- Mileage (1.59%): Slight influence on the car's price.\n",
    "\n",
    "**Brand and Location Influence:**\n",
    "\n",
    "- Transmission Manual (0.57%): Slightly affects prices.\n",
    "- Brand TATA (0.53%) and Brand HONDA (0.37%): Notable impact among brands.\n",
    "- Location Kolkata (0.53%) and Location Hyderabad (0.31%): Locations with significant influence.\n",
    "\n",
    "**Minor Influences:**\n",
    "\n",
    "- Seats (0.38%): Small effect on price.\n",
    "- Fuel Type Diesel (0.21%): Minor effect among fuel types.\n",
    "\n",
    "**Least Important Features:**\n",
    "\n",
    "- Some brands and locations have minimal or no impact (e.g., Brand Lamborghini, Brand Force, Brand Datsun, Brand SMART, Brand Bentley).\n",
    "\n",
    "### Summary:\n",
    "\n",
    "- **Power** and **CarAge** are the primary determinants of car prices.\n",
    "- **Engine size** and **mileage** have moderate effects.\n",
    "- **Brand** and **location** show varying levels of influence.\n",
    "- **Transmission type** and **fuel type** contribute but are less significant.\n",
    "- **Several features have negligible impact**, suggesting potential for simplifying the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sw0dMSgetOS_"
   },
   "source": [
    "### **Hyperparameter Tuning: Decision Tree**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 338,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-37 {color: black;background-color: white;}#sk-container-id-37 pre{padding: 0;}#sk-container-id-37 div.sk-toggleable {background-color: white;}#sk-container-id-37 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-37 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-37 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-37 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-37 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-37 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-37 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-37 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-37 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-37 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-37 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-37 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-37 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-37 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-37 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-37 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-37 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-37 div.sk-item {position: relative;z-index: 1;}#sk-container-id-37 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-37 div.sk-item::before, #sk-container-id-37 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-37 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-37 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-37 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-37 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-37 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-37 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-37 div.sk-label-container {text-align: center;}#sk-container-id-37 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-37 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-37\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeRegressor(max_depth=15, min_samples_leaf=3, min_samples_split=30,\n",
       "                      random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-37\" type=\"checkbox\" checked><label for=\"sk-estimator-id-37\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeRegressor</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeRegressor(max_depth=15, min_samples_leaf=3, min_samples_split=30,\n",
       "                      random_state=1)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeRegressor(max_depth=15, min_samples_leaf=3, min_samples_split=30,\n",
       "                      random_state=1)"
      ]
     },
     "execution_count": 338,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "# Choose the type of estimator\n",
    "dtree_tuned = DecisionTreeRegressor(random_state=1)\n",
    "\n",
    "# Grid of parameters to choose from\n",
    "parameters = {\n",
    "    'max_depth': [10,15,20,25,30],\n",
    "    'min_samples_leaf': [3, 15,30],\n",
    "    'min_samples_split': [15,30,35,40,50],\n",
    "}\n",
    "\n",
    "# Type of scoring used to compare parameter combinations\n",
    "scorer = 'neg_mean_squared_error'\n",
    "\n",
    "# Run the grid search\n",
    "grid_obj = GridSearchCV(estimator=dtree_tuned, param_grid=parameters, scoring=scorer, cv=5)\n",
    "grid_obj = grid_obj.fit(X_train, y_train['price_log'])\n",
    "\n",
    "# Set the model to the best combination of parameters\n",
    "dtree_tuned = grid_obj.best_estimator_\n",
    "\n",
    "# Fit the best algorithm to the data\n",
    "dtree_tuned.fit(X_train, y_train['price_log'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 339,
   "metadata": {
    "id": "hctfJIAXtOS_"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-sqaure on training set :  0.9016184279074143\n",
      "R-square on test set :  0.822259111769305\n",
      "RMSE on training set :  3.5043554228410305\n",
      "RMSE on test set :  4.698650156870669\n"
     ]
    }
   ],
   "source": [
    "# Get score of the dtree_tuned\n",
    "dtree_tuned_score = get_model_score(dtree_tuned)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "UsGmvq1StOS_"
   },
   "source": [
    "### Observations from Tuned Decision Tree Regression Results\n",
    "\n",
    "**Model Fit:**\n",
    "\n",
    "- R-squared: **Training:** 0.9016, **Test:** 0.8223\n",
    "- RMSE: **Training:** 3.5044, **Test:** 4.6987\n",
    "\n",
    "**Generalization:**\n",
    "\n",
    "- Improved generalization with smaller differences in R-squared values between training and test sets.\n",
    "- Reduced RMSE on the test set indicates better predictive performance.\n",
    "\n",
    "**Prediction Accuracy:**\n",
    "\n",
    "- Achieves a good balance between fit and generalization, with improved test set accuracy and reasonable training set performance.\n",
    "\n",
    "**Conclusion:**\n",
    "\n",
    "- Hyperparameter tuning enhanced the decision tree's performance, leading to better generalization and prediction accuracy.\n",
    "- The model now better predicts unseen data, reducing overfitting observed in the initial model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5jssEF5eIrdH"
   },
   "source": [
    "**Feature Importance**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 340,
   "metadata": {
    "id": "OdzQWq8WtOTA"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                Imp\n",
      "Power                      0.644293\n",
      "CarAge                     0.240857\n",
      "Engine                     0.050725\n",
      "Mileage                    0.008531\n",
      "Transmission_Manual        0.005594\n",
      "kilometers_driven_log      0.005400\n",
      "Brand_HONDA                0.004790\n",
      "Brand_TATA                 0.004703\n",
      "Location_Kolkata           0.004470\n",
      "Brand_MAHINDRA             0.003475\n",
      "Brand_AUDI                 0.003373\n",
      "Brand_MINI                 0.003362\n",
      "Fuel_Type_Diesel           0.002801\n",
      "Brand_SKODA                0.002654\n",
      "Brand_TOYOTA               0.002017\n",
      "Seats                      0.001577\n",
      "Location_Hyderabad         0.001564\n",
      "Location_Coimbatore        0.001443\n",
      "Brand_MARUTI               0.001103\n",
      "Brand_HYUNDAI              0.001062\n",
      "Location_Jaipur            0.000772\n",
      "Location_Mumbai            0.000653\n",
      "Brand_PORSCHE              0.000612\n",
      "Brand_LAND                 0.000603\n",
      "Brand_CHEVROLET            0.000597\n",
      "Brand_FORD                 0.000488\n",
      "Location_Chennai           0.000328\n",
      "Owner_Type_Second          0.000322\n",
      "Brand_JEEP                 0.000310\n",
      "Location_Delhi             0.000307\n",
      "Location_Bangalore         0.000240\n",
      "Owner_Type_Third           0.000178\n",
      "Location_Pune              0.000176\n",
      "Brand_VOLKSWAGEN           0.000160\n",
      "Location_Kochi             0.000142\n",
      "Brand_RENAULT              0.000118\n",
      "Brand_VOLVO                0.000078\n",
      "Brand_MITSUBISHI           0.000066\n",
      "Brand_NISSAN               0.000056\n",
      "Brand_LAMBORGHINI          0.000000\n",
      "Brand_JAGUAR               0.000000\n",
      "Brand_ISUZU                0.000000\n",
      "Brand_FORCE                0.000000\n",
      "Brand_MERCEDES-BENZ        0.000000\n",
      "Brand_FIAT                 0.000000\n",
      "Brand_DATSUN               0.000000\n",
      "Brand_BENTLEY              0.000000\n",
      "Owner_Type_Fourth & Above  0.000000\n",
      "Brand_SMART                0.000000\n",
      "Fuel_Type_Petrol           0.000000\n",
      "Fuel_Type_LPG              0.000000\n",
      "Fuel_Type_Electric         0.000000\n",
      "Brand_BMW                  0.000000\n"
     ]
    }
   ],
   "source": [
    "# Print important features of tuned decision tree similar to decision trees\n",
    "print(pd.DataFrame(dtree_tuned.feature_importances_, columns = [\"Imp\"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-r8AR_VotOTB"
   },
   "source": [
    "### Observations and Insights from Tuned Decision Tree Feature Importances\n",
    "\n",
    "**Top Features:**\n",
    "\n",
    "- Power (64.43%): The most influential factor in predicting car prices.\n",
    "- CarAge (24.09%): Significant impact, reflecting the car's age effect on price.\n",
    "- Engine (5.07%): Important but less influential than Power and CarAge.\n",
    "\n",
    "**Moderate Importance:**\n",
    "\n",
    "- Mileage (0.85%): Slight influence on the car's price.\n",
    "- Transmission Manual (0.56%): Slightly affects prices.\n",
    "- Kilometers Driven (log) (0.54%): Moderately affects prices.\n",
    "\n",
    "**Brand and Location Influence:**\n",
    "\n",
    "- Brand HONDA (0.48%) and Brand TATA (0.47%): Notable impact among brands.\n",
    "- Location Kolkata (0.45%) and Location Hyderabad (0.16%): Significant location effects.\n",
    "\n",
    "**Other Notable Features:**\n",
    "\n",
    "- Seats (0.16%): Minor influence on price.\n",
    "- Fuel Type Diesel (0.28%): Small effect among fuel types.\n",
    "\n",
    "**Least Important Features:**\n",
    "\n",
    "- Some brands and locations have minimal or no impact (e.g., Brand Lamborghini, Brand Force, Brand Datsun, Brand SMART, Brand Bentley, Fuel Type Petrol, Fuel Type LPG, Fuel Type Electric).\n",
    "\n",
    "### Summary:\n",
    "\n",
    "- **Power** and **CarAge** remain the primary determinants of car prices.\n",
    "- **Engine size** and **mileage** have moderate effects.\n",
    "- **Brand** and **location** continue to show varying levels of influence on prices.\n",
    "- **Transmission type** and **ownership history** contribute but are less significant.\n",
    "- **Several features have negligible impact**, suggesting potential for model simplification."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "18uxHTy2tOTB"
   },
   "source": [
    "### **Hyperparameter Tuning: Random Forest**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 341,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-38 {color: black;background-color: white;}#sk-container-id-38 pre{padding: 0;}#sk-container-id-38 div.sk-toggleable {background-color: white;}#sk-container-id-38 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-38 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-38 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-38 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-38 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-38 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-38 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-38 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-38 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-38 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-38 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-38 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-38 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-38 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-38 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-38 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-38 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-38 div.sk-item {position: relative;z-index: 1;}#sk-container-id-38 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-38 div.sk-item::before, #sk-container-id-38 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-38 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-38 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-38 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-38 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-38 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-38 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-38 div.sk-label-container {text-align: center;}#sk-container-id-38 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-38 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-38\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor(max_depth=20, min_samples_split=10, random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-38\" type=\"checkbox\" checked><label for=\"sk-estimator-id-38\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor(max_depth=20, min_samples_split=10, random_state=1)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestRegressor(max_depth=20, min_samples_split=10, random_state=1)"
      ]
     },
     "execution_count": 341,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "\n",
    "# Choose the type of Regressor\n",
    "rfr_tuned = RandomForestRegressor(random_state=1)\n",
    "\n",
    "# Define the parameters for Randomized Search to choose from\n",
    "parameters = {\n",
    "    'n_estimators': [100, 200],\n",
    "    'max_depth': [10, 20],\n",
    "    'min_samples_split': [2, 10],\n",
    "    'min_samples_leaf': [1, 5]\n",
    "}\n",
    "\n",
    "# Type of scoring used to compare parameter combinations\n",
    "scorer = 'neg_mean_squared_error'\n",
    "\n",
    "# Run the randomized search\n",
    "grid_obj = RandomizedSearchCV(estimator=rfr_tuned, param_distributions=parameters, scoring=scorer, cv=5, n_iter=10, n_jobs=-1, random_state=1)\n",
    "grid_obj = grid_obj.fit(X_train, y_train['price_log'])\n",
    "\n",
    "# Set the model to the best combination of parameters\n",
    "rfr_tuned = grid_obj.best_estimator_\n",
    "\n",
    "# Fit the best algorithm to the data\n",
    "rfr_tuned.fit(X_train, y_train['price_log'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 342,
   "metadata": {
    "id": "HSBtYgpctOTC"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-sqaure on training set :  0.9434038075336219\n",
      "R-square on test set :  0.8694539417894435\n",
      "RMSE on training set :  2.6579381798820427\n",
      "RMSE on test set :  4.026811672041718\n",
      "[0.9434038075336219, 0.8694539417894435, 2.6579381798820427, 4.026811672041718]\n"
     ]
    }
   ],
   "source": [
    "# Get score of the model\n",
    "rfr_tuned_score = get_model_score(rfr_tuned)\n",
    "print(rfr_tuned_score)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "a1WHqIX9tOTC"
   },
   "source": [
    "### Observations from Tuned Random Forest Regression Results\n",
    "\n",
    "**Model Fit:**\n",
    "\n",
    "- R-squared: **Training:** 0.9434, **Test:** 0.8695\n",
    "- RMSE: **Training:** 2.6579, **Test:** 4.0268\n",
    "\n",
    "**Generalization:**\n",
    "\n",
    "- Excellent generalization with minimal overfitting, indicated by high R-squared and low RMSE values on both sets.\n",
    "\n",
    "**Prediction Accuracy:**\n",
    "\n",
    "- High accuracy with low RMSE, confirming good performance on unseen data.\n",
    "\n",
    "**Conclusion:**\n",
    "\n",
    "- Hyperparameter tuning significantly improved the model's performance, resulting in strong predictive capabilities."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ItsgSUyiIrdI"
   },
   "source": [
    "**Feature Importance**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 343,
   "metadata": {
    "id": "9khvM2ZhtOTC"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                Imp\n",
      "Power                      0.634340\n",
      "CarAge                     0.238522\n",
      "Engine                     0.037414\n",
      "kilometers_driven_log      0.014801\n",
      "Mileage                    0.013994\n",
      "Location_Kolkata           0.004980\n",
      "Brand_TATA                 0.004784\n",
      "Transmission_Manual        0.004640\n",
      "Seats                      0.003548\n",
      "Brand_HONDA                0.003483\n",
      "Brand_MAHINDRA             0.002972\n",
      "Brand_MINI                 0.002958\n",
      "Location_Hyderabad         0.002677\n",
      "Brand_MERCEDES-BENZ        0.002408\n",
      "Brand_AUDI                 0.002117\n",
      "Location_Coimbatore        0.002092\n",
      "Brand_SKODA                0.002051\n",
      "Brand_TOYOTA               0.001826\n",
      "Brand_LAND                 0.001627\n",
      "Fuel_Type_Diesel           0.001494\n",
      "Brand_VOLKSWAGEN           0.001420\n",
      "Brand_CHEVROLET            0.001357\n",
      "Brand_HYUNDAI              0.001300\n",
      "Location_Bangalore         0.001247\n",
      "Fuel_Type_Petrol           0.001239\n",
      "Location_Delhi             0.001052\n",
      "Location_Mumbai            0.000969\n",
      "Location_Jaipur            0.000922\n",
      "Brand_MARUTI               0.000908\n",
      "Owner_Type_Second          0.000840\n",
      "Location_Pune              0.000823\n",
      "Brand_PORSCHE              0.000677\n",
      "Location_Chennai           0.000673\n",
      "Owner_Type_Third           0.000644\n",
      "Location_Kochi             0.000638\n",
      "Brand_BMW                  0.000595\n",
      "Brand_FORD                 0.000486\n",
      "Fuel_Type_Electric         0.000262\n",
      "Brand_RENAULT              0.000236\n",
      "Brand_FIAT                 0.000232\n",
      "Brand_JEEP                 0.000168\n",
      "Brand_NISSAN               0.000147\n",
      "Brand_JAGUAR               0.000118\n",
      "Brand_MITSUBISHI           0.000111\n",
      "Brand_VOLVO                0.000064\n",
      "Brand_LAMBORGHINI          0.000061\n",
      "Brand_DATSUN               0.000031\n",
      "Owner_Type_Fourth & Above  0.000030\n",
      "Brand_SMART                0.000009\n",
      "Fuel_Type_LPG              0.000007\n",
      "Brand_ISUZU                0.000006\n",
      "Brand_FORCE                0.000002\n",
      "Brand_BENTLEY              0.000000\n"
     ]
    }
   ],
   "source": [
    "# Print important features of tuned random forest\n",
    "feature_importances = pd.DataFrame(rfr_tuned.feature_importances_, columns=[\"Imp\"], index=X_train.columns).sort_values(by='Imp', ascending=False)\n",
    "print(feature_importances)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PBoHEXnjtOTC"
   },
   "source": [
    "### Observations and Insights from Tuned Random Forest Feature Importances\n",
    "\n",
    "**Top Features:**\n",
    "\n",
    "- Power (63.43%): The most critical factor influencing car prices.\n",
    "- CarAge (23.85%): Significant impact on price.\n",
    "- Engine (3.74%): Important, but less influential than Power and CarAge.\n",
    "\n",
    "**Moderate Importance:**\n",
    "\n",
    "- Kilometers Driven (log) (1.48%): Moderate effect on prices.\n",
    "- Mileage (1.40%): Slight influence on car prices.\n",
    "\n",
    "**Brand and Location Influence:**\n",
    "\n",
    "- Location Kolkata (0.50%) and Location Hyderabad (0.27%): Notable location effects.\n",
    "- Brand TATA (0.48%) and Brand HONDA (0.35%): Significant brand impact.\n",
    "\n",
    "**Transmission and Ownership:**\n",
    "\n",
    "- Transmission Manual (0.46%): Slightly affects prices.\n",
    "- Owner Type Second (0.08%): Minor impact from previous ownership.\n",
    "\n",
    "**Other Notable Features:**\n",
    "\n",
    "- Seats (0.35%): Minor influence on price.\n",
    "- Fuel Type Diesel (0.15%): Small effect among fuel types.\n",
    "\n",
    "**Least Important Features:**\n",
    "\n",
    "- Some brands and locations have minimal or no impact (e.g., Brand Lamborghini, Brand Force, Brand Datsun, Brand SMART, Brand Bentley, Fuel Type LPG).\n",
    "\n",
    "### Summary:\n",
    "\n",
    "- **Power** and **CarAge** are the primary determinants of car prices.\n",
    "- **Engine size** and **mileage** have moderate effects.\n",
    "- **Brand** and **location** show varying levels of influence.\n",
    "- **Transmission type** and **ownership history** are less significant.\n",
    "- **Several features have negligible impact**, suggesting potential for model simplification."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 344,
   "metadata": {
    "id": "DCInk4Y8tOTC"
   },
   "outputs": [],
   "source": [
    "# Defining list of models you have trained\n",
    "models = [lr, dtree, ridge, rfr, dtree_tuned, rfr_tuned]\n",
    "\n",
    "# Defining empty lists to add train and test results\n",
    "r2_train = []\n",
    "r2_test = []\n",
    "rmse_train = []\n",
    "rmse_test = []\n",
    "\n",
    "# Looping through all the models to get the rmse and r2 scores\n",
    "for model in models:\n",
    "    \n",
    "    # Accuracy score\n",
    "    j = get_model_score(model, False)\n",
    "    \n",
    "    r2_train.append(j[0])\n",
    "    r2_test.append(j[1])\n",
    "    rmse_train.append(j[2])\n",
    "    rmse_test.append(j[3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 345,
   "metadata": {
    "id": "zuLokC7xtOTD"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Train_r2</th>\n",
       "      <th>Test_r2</th>\n",
       "      <th>Train_RMSE</th>\n",
       "      <th>Test_RMSE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Linear Regression</td>\n",
       "      <td>0.860693</td>\n",
       "      <td>0.867104</td>\n",
       "      <td>4.170016</td>\n",
       "      <td>4.062897</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Decision Tree</td>\n",
       "      <td>0.999997</td>\n",
       "      <td>0.798364</td>\n",
       "      <td>0.020693</td>\n",
       "      <td>5.004537</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Ridge Regression</td>\n",
       "      <td>0.857324</td>\n",
       "      <td>0.865241</td>\n",
       "      <td>4.220144</td>\n",
       "      <td>4.091279</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.975233</td>\n",
       "      <td>0.874580</td>\n",
       "      <td>1.758292</td>\n",
       "      <td>3.946964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Tuned Decision Tree</td>\n",
       "      <td>0.901618</td>\n",
       "      <td>0.822259</td>\n",
       "      <td>3.504355</td>\n",
       "      <td>4.698650</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Tuned Random Forest</td>\n",
       "      <td>0.943404</td>\n",
       "      <td>0.869454</td>\n",
       "      <td>2.657938</td>\n",
       "      <td>4.026812</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Model  Train_r2   Test_r2  Train_RMSE  Test_RMSE\n",
       "0    Linear Regression  0.860693  0.867104    4.170016   4.062897\n",
       "1        Decision Tree  0.999997  0.798364    0.020693   5.004537\n",
       "2     Ridge Regression  0.857324  0.865241    4.220144   4.091279\n",
       "3        Random Forest  0.975233  0.874580    1.758292   3.946964\n",
       "4  Tuned Decision Tree  0.901618  0.822259    3.504355   4.698650\n",
       "5  Tuned Random Forest  0.943404  0.869454    2.657938   4.026812"
      ]
     },
     "execution_count": 345,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Creating a DataFrame to compare the models\n",
    "comparison_frame = pd.DataFrame({\n",
    "    'Model': ['Linear Regression', 'Decision Tree', 'Ridge Regression', 'Random Forest', 'Tuned Decision Tree', 'Tuned Random Forest'],\n",
    "    'Train_r2': r2_train,\n",
    "    'Test_r2': r2_test,\n",
    "    'Train_RMSE': rmse_train,\n",
    "    'Test_RMSE': rmse_test\n",
    "})\n",
    "\n",
    "# Display the comparison frame\n",
    "comparison_frame"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TZrq2E9VtOTD"
   },
   "source": [
    "**Observations:**\n",
    "\n",
    "- **Linear Regression** model shows an R-squared of approximately 0.861 on the training set and 0.867 on the test set, with RMSE values around 4.17 and 4.06 respectively.\n",
    "- **Ridge Regression** performs similarly to Linear Regression, with slightly lower R-squared and slightly higher RMSE values.\n",
    "- **Decision Tree** model has a very high R-squared on the training set (0.999997), indicating overfitting, as the R-squared drops significantly to 0.798 on the test set. The RMSE values further confirm this with a drastic increase from 0.02 (training) to 5.00 (test).\n",
    "- **Random Forest** models (both tuned and untuned) show strong performance, with high R-squared values and low RMSE values on both training and test sets, indicating good generalization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "58KMVhO_tOTD"
   },
   "source": [
    "### **Insights**\n",
    "\n",
    "**Refined insights**:\n",
    "\n",
    "- The models based on ensemble methods (Random Forest and Tuned Random Forest) perform better compared to single decision tree models, showing high predictive accuracy and better generalization.\n",
    "- Decision Tree overfits the training data and performs poorly on the test set, making it less reliable for prediction.\n",
    "- Linear Regression and Ridge Regression are decent but not as powerful as the ensemble methods.\n",
    "\n",
    "**Comparison of various techniques and their relative performance**:\n",
    "\n",
    "- Random Forest models outperform the rest, providing a balance between accuracy and generalization.\n",
    "- Decision Trees tend to overfit without tuning.\n",
    "- Regularized methods like Ridge Regression perform slightly better than standard Linear Regression.\n",
    "\n",
    "**Proposal for the final solution design**:\n",
    "\n",
    "- ***Tuned Random Forest*** should be adopted as the final model for predicting car prices. It demonstrates the highest accuracy and best generalization capability among the models evaluated.\n",
    "- ***Tuned Random Forest*** strikes a balance between complexity and performance, making it the most suitable model for this problem."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "a1WHqIX9tOTC",
    "PBoHEXnjtOTC",
    "TZrq2E9VtOTD"
   ],
   "name": "Reference_Notebook_Milestone_2_Regression.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
