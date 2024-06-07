RRT Research Assignment 2024
Created by Tanvi Parulekar 2024

This program has the main file, robotFollowingRRT.py.
The program includes
-an ADT of an RRT: sets up the tree, random sampling, nearest node, obstacle checking requirements of an RRT algorithm, as well as tracing the path
-a function that plots the traced path
-class Robot that simulates a 10m by 15m robot following the path found by the RRT
-Three algorithms run by functions: basic RRT, RRT Connect pioneered by J. Kuffner, and the Greedy RRT proposed by Yingrui Xie et. al.
-Measures of algorithms: time decorator to measure time the algorithm took; path distance measured during trace path function; number of sampled nodes

Test results are saved in different folders.
The three algorithms were tested against four different maps taken from another paper (simple, complex, indoors, narrow passages).
