import win32com.client
import pythoncom
from auto_parse import parse_file as auto_parse_file
import time


readerType = 26 
readerSerialNumber = ""  


def get_reader_status():
    """
    This function is used to get the status of the reader.
    """

    gen5App = win32com.client.Dispatch("Gen5.Application", pythoncom.CoInitialize())
    gen5App.ConfigureUSBReader(readerType, readerSerialNumber)

    # Test the communication with the reader
    max_try = 50
    current_try = 0

    for _ in range(max_try):
        # result = self.gen5App.TestReaderCommunication
        status = gen5App.GetReaderStatus
        communication_status = gen5App.TestReaderCommunication

        current_try += 1
        if status == 0:
            break
        else:
            last_error = gen5App.TestReaderCommunication
            print(f"Current reader status: {status}; Communication status: {communication_status}; Error encountered: {last_error}" )
        time.sleep(2)


    pythoncom.CoUninitialize()


    if current_try >= max_try:
        print("Reader not ready.")
        return False, last_error


    return True, "Reader ready."




def carrier_in():
    """
    This function is used to move the carrier into the reader.
    """

    is_ready, error_msg  = get_reader_status()


    gen5App = win32com.client.Dispatch("Gen5.Application", pythoncom.CoInitialize())
    gen5App.ConfigureUSBReader(readerType, readerSerialNumber)


    if not is_ready:
        return False, error_msg

    gen5App.CarrierIn()

    pythoncom.CoUninitialize()

    return True, "Carrier in successful."


def carrier_out():
    """
    This function is used to move the carrier out of the reader.
    """
    is_ready, error_msg  = get_reader_status()

    gen5App = win32com.client.Dispatch("Gen5.Application", pythoncom.CoInitialize())
    gen5App.ConfigureUSBReader(readerType, readerSerialNumber)


    if not is_ready:
        return False, error_msg


    gen5App.CarrierOut()
    
    pythoncom.CoUninitialize()
    
    return True, "Carrier out successful."


def read_plate(protocol_path: str, experiment_path: str, csv_path: str):

    """
    This function is used to read a plate using the Gen5 software.

    Args:
        protocol_path (str): The path to the protocol file.
        experiment_path (str): The path to save the experiment file.
        csv_path (str): The path to save the CSV file.
    """

    try:

        is_ready, error_msg  = get_reader_status()

        gen5App = win32com.client.Dispatch("Gen5.Application", pythoncom.CoInitialize())
        gen5App.ConfigureUSBReader(readerType, readerSerialNumber)


        if not is_ready:
            return False, error_msg

        # Create a new experiment using the specified protocol file
        experiment = gen5App.NewExperiment(protocol_path)

        # Get the first plate from the experiment
        plate = experiment.Plates.GetPlate(1)

        # Start the plate read
        readMonitor = plate.StartReadEx(False)

        # Monitor the read status
        while readMonitor.ReadInProgress:
            # Wait or perform other tasks
            pass


        # Save the experiment to the specified file path
        experiment.SaveAs(experiment_path)

        # Export the reading results as CSV
        plate.FileExport(csv_path)

        experiment.Close()

        pythoncom.CoUninitialize()

        return True, "Plate read successful."

    except Exception as e:
        return False, e
    

def parse_file(csv_path: str):
    parsed_info = auto_parse_file(csv_path)
    return parsed_info