from the_matrix import the_matrix

the_matrix_generator = the_matrix(generation_model_path="./models/stage2", streaming_model_path="./models/stage3")
the_matrix_generator.generate(
    prompt="In a barren desert, a white SUV is driving. From an overhead panoramic shot, the vehicle has blue and red stripe decorations on its body, \
            and there is a black spoiler at the rear. It is traversing through sand dunes and shrubs, kicking up a cloud of dust. In the distance, \
            undulating mountains can be seen, with a sky of deep blue and a few white clouds floating by. There are also some green plants in the distance.",
    length=8,
    output_folder="./",
    # control_signal="D,D,D,D,D,DR,DR,DR,DR,DR,D,D,D,D,D,D,D,D,D,DL,DL,DL,DL,DL,DL,D,D,D,D,D,D,D,D"
)