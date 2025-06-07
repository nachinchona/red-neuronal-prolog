:- dynamic neuron/4.
% neuron(ID, Function-Arg, Connections, Output)
:- dynamic input_buffer/3.
% input_buffer(NeuronID, ListOfWeightedInputs, Counter)
:- include('leqneurons.pl').
:- use_module(library(dcg/basics)).

neurons_per_layer(13).

reset:-
    retractall(neuron(_,_,_,_)),
    retractall(input_buffer(_,_,_)).

run_training(N, TrainedFile) :-
    phrase_from_file(data(EntrySets), "wine.trainingset"),
    numlist(1, N, Iterations),
    train_n_times(Iterations, EntrySets),
    save_neurons(TrainedFile).

run_testing:-
    phrase_from_file(data(EntrySets), "wine.test"),
    length(EntrySets,N),
    test(EntrySets,0,N).

%DCG para parsear datos a una lista
%------------------------------------------------------------------------
data(_) --> eos.
data([InputList|NextLists]) --> line(InputList), data(NextLists).

line([N|Ns]) --> integer(N), rest_of_line(Ns), eol. 

rest_of_line([N|Ns]) --> ",", number_or_float(N), rest_of_line(Ns).
rest_of_line([]) --> [].

number_or_float(N) --> float(N), !.
number_or_float(N) --> integer(N).
%------------------------------------------------------------------------

%Permite repetir entrenamiento N veces y notifica por pantalla
train_n_times([], _).
train_n_times([I|Rest], EntrySets) :-
    train(EntrySets),
    format("Training iteration ~d~n", [I]),
    train_n_times(Rest, EntrySets).

%Guarda las neuronas entrenadas
save_neurons(File) :-
    tell(File),
    listing(neuron/4),
    listing(input_buffer/3),
    told.

train([]).
train([[ClassNum|EntrySet]|NextEntrySets]):-
    load_first_layer(EntrySet,1),
    forall((neuron(Id, Func-Arg, Connec, Output), Connec \= []),
	   (input_buffer(Id,EntryValueList,_),
	    sum_list(EntryValueList,EntryValue),
	    update_weights(ClassNum,Connec,EntryValue,NewConnec),
	    retractall(neuron(Id, Func-Arg, Connec, Output)),
	    assertz(neuron(Id, Func-Arg, NewConnec, Output)))),
    restart,
    train(NextEntrySets).

test([],CorrectAnswers,Total):-
    IncorrectAnswers is Total - CorrectAnswers,
    format("Finished. Correct: ~w, Incorrect: ~w, Total: ~w~n",[CorrectAnswers, IncorrectAnswers, Total]).

test([[ClassNum|EntrySet]|NextEntrySets], CorrectAnswers, T) :-
    load_first_layer(EntrySet, 1),
    ((neuron((N,ClassNum), _, [], 1.0),
      forall((neuron(ID, _, [], Output), ID \= (N,ClassNum)),
	      Output = 0.0))
    ->  C1 is CorrectAnswers + 1
    ;   C1 = CorrectAnswers
    ),
    restart,
    test(NextEntrySets, C1, T).

%Carga los input buffers de la primera capa de neuronas
load_first_layer([],_).
load_first_layer([H|T],C):-
    Id = (1,C),
    retract(input_buffer(Id, _, _)),
    neurons_per_layer(X),
    X1 is X-1,
    assertz(input_buffer(Id,[],X1)),
    feed_input(Id,H),
    C1 is C+1,
    load_first_layer(T,C1).

%Actualiza los pesos de los enlaces a neuronas
%Asume solo 2 capas, por la formula utilizada
update_weights(_,[],_,[]).
update_weights(ClassNum, [N-Weight|Weights], EntryValue, [N-NewWeight|NewWeights]) :-
     neuron(N, _, _, Output),
    (   N = (_, ClassNum)
    ->  ExpectedOutput = 1
    ;   ExpectedOutput = 0
    ),
    NewWeight is Weight + 0.01*(ExpectedOutput - Output)*EntryValue,
    update_weights(ClassNum, Weights, EntryValue, NewWeights).

% Dar entrada a neurona y activarla si ya tiene todas
feed_input(ID, Value) :-
    input_buffer(ID,Inputs,C),
    retract(input_buffer(ID, Inputs, C)),
    C1 is C+1,
    assertz(input_buffer(ID, [Value|Inputs],C1)),
    neurons_per_layer(C1) -> activate(ID) ; true.

% Activar neurona 
activate(ID) :-
    input_buffer(ID, Inputs, _),
    sum_list(Inputs, TotalInput),
    neuron(ID, Func-Arg, Connections, _),
    call(Func, TotalInput, Arg, Output),
    retract(neuron(ID, Func-Arg, Connections, _)),
    assertz(neuron(ID, Func-Arg, Connections, Output)),
    propagate(ID, Output, Connections).

% Propagar resultados a vecinos
propagate(_, _, []).
propagate(ID, Output, [TargetID-Weight | Rest]) :-
    Weighted is Output * Weight,
    feed_input(TargetID, Weighted),
    propagate(ID, Output, Rest).

%Reestablece todos los input buffers
restart :-
    forall(input_buffer(Id, Inputs, C),
	   (retract(input_buffer(Id,Inputs,C)),
	    assertz(input_buffer(Id,[],0)))).

%Funcion usada por las neuronas
geq(X,Y,1.0):- X >= Y.
geq(X,Y,0.0):- X < Y.

%Calculamos diferente para positivos y negativos para evitar float overflow
%El 2do argumento es para mantener interfaz que permite usar cualquier funcion
sigmoid(X, _, Y) :-
    (   X >= 0
    ->  ExpNegX is exp(-X),
        Y is 1 / (1 + ExpNegX)
    ;   ExpX is exp(X),
        Y is ExpX / (1 + ExpX)
    ).
