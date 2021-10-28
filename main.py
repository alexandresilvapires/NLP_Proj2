import utils
import metrics
import model


# for debug purposes
def main():
    # DOWNLOAD RESOURCES
    metrics.download_resources()

    # TESTING METRICS
    print('Second column array:')
    print(metrics.text_to_column_array("./trainWithoutDev.txt", 2)) # This prints \n if column=2
    print('\n')


    # TESTING MODEL
    print("Naive Bayes accuracy:")
    model.run_model_test("./dev.txt", "./trainWithoutDev.txt")
    print('\n')


    # TESTING UTILS
    print("DATAFRAME:")
    dataframe = utils.file_to_dataframe("./trainWithoutDev.txt")
    print(dataframe, '\n')

    print("HISTORY LINES:")
    history_lines = utils.get_fixed_column_lines(dataframe, "Category", "HISTORY")
    print(history_lines, '\n')

    print("THIRD LINE:")
    third_line = utils.get_row_by_number(dataframe, 2)
    print(third_line, '\n')

    # END OF TESTING


if __name__ == '__main__':
    main()