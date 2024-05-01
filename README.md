# Movie_recommendation_system
Build a movie recommendation system based on genre and title, using machine learning


## Dataset

Here is part of the dataset:
| "	movie_id	title	genres" |
| --- |
"0	1	Toy Story (1995)	Animation|Children's|Comedy"
"1	2	Jumanji (1995)	Adventure|Children's|Fantasy"
"2	3	Grumpier Old Men (1995)	Comedy|Romance"
"3	4	Waiting to Exhale (1995)	Comedy|Drama"
"4	5	Father of the Bride Part II (1995)	Comedy"
"5	6	Heat (1995)	Action|Crime|Thriller"
"6	7	Sabrina (1995)	Comedy|Romance"
"7	8	Tom and Huck (1995)	Adventure|Children's"
"8	9	Sudden Death (1995)	Action"
"9	10	GoldenEye (1995)	Action|Adventure|Thriller"
"10	11	American President"
"11	12	Dracula: Dead and Loving It (1995)	Comedy|Horror"
"12	13	Balto (1995)	Animation|Children's"
"13	14	Nixon (1995)	Drama"
"14	15	Cutthroat Island (1995)	Action|Adventure|Romance"
"15	16	Casino (1995)	Drama|Thriller"
"16	17	Sense and Sensibility (1995)	Drama|Romance"
"17	18	Four Rooms (1995)	Thriller"
"18	19	Ace Ventura: When Nature Calls (1995)	Comedy"
"19	20	Money Train (1995)	Action"
"20	21	Get Shorty (1995)	Action|Comedy|Drama"
"21	22	Copycat (1995)	Crime|Drama|Thriller"
"22	23	Assassins (1995)	Thriller"
"23	24	Powder (1995)	Drama|Sci-Fi"
"24	25	Leaving Las Vegas (1995)	Drama|Romance"


## Pre-processing

Split values in the "genres" column (by "|")


## Training

* Create a TfidfVectorizer for the "genres" column

Let's see the matrix:

![image](https://github.com/T-KIEU/Movie_recommendation_system/assets/100022674/9eb0ea40-c1eb-4e40-9cb8-02aa1f3e3efa)

Each movie is represented by a vector. <br />
<br />

* To estimate the similiarity between two movies, we must calculate the cosine similiarity of the angle between these two vectors
* Create a recommendation function
* Apply the recommendation function to a movie to see other movies recommended


