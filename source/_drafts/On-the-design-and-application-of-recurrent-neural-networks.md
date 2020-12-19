---
title: On the design and application of recurrent neural networks
url: 1101.html
id: 1101
categories:
  - Uncategorized
tags:
---

We have so many recurrent neural nets these days. What is the landscape of RNN? What is the interleaving part between RNN and other methods? Different types: the main idea is to increase the capacity of memory. LSTM, Gated Recurrent Unit,  Memory Nets and Neural Turing Machine all have mechanisms to access excess memory cell so that they can embrace enough knowledge and information to achieve the proposed goal. Truncated Back propagation A little bit history RNN has been raised by List of the inventors：

1.  vanilla RNN
2.  LSTM
3.  GRU

  I guess during the development, people got these interesting and powerful findings with fortune. Oriol Vinyals, one of Innovators Under 35, 2016, by MIT Technology Review said “I remember it so well,”  “I changed a single line of code: instead of translating from French, I changed my code to input an image instead.” Then he found the idea worked. Instead of just a common boring description for the picture he send into the algorithm, the result was so much sophisticated. Before doing research, he was a electronic-game athlete as a adolescent. When he was in UC Berkeley, together with Pieter Abeel and other team members, he build a AI program to play starcraft. Another famous game named Counter-Striker, video prediction through RNN   A module view for applying RNN: better interface   We try to visualize the landscape for this area to help us get better sense about it.