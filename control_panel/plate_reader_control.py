import time
import pyautogui as pa

print(pa.size())

# pa.moveTo(100,100, duration=0.1)
print(pa.position())

def click_taskbar_app():
    """
    click the taskbar app
    """
    pa.moveTo(606, 1057, duration=0.1)
    pa.click()
    return True


def click_instrument_control():
    """
    click the instrument control button
    """
    pa.moveTo(762, 601, duration=0.1)
    pa.click()
    return True

def skip_saving_experiment_file():
    """
    skip saving the expierment file after reading (we will save excel)
    """
    pa.hotkey('esc')
    return True

def proceed_saving_excel():
    """
    proceed to saving the excel file
    """
    pa.press("enter")
    return True



def init():
    click_taskbar_app()
    time.sleep(0.1)
    click_instrument_control()


def plate_out():
    """
    click the instrument plate out button
    """
    pa.moveTo(1113, 529, duration=0.1)
    pa.click()
    return True

def click_protocal_tab():
    """
    navigate to the protocol tab and click
    """
    pa.moveTo(764, 532, duration=0.1)
    pa.click()
    return True

def click_one_glo_protocal():
    """
    click the one glo protocal button.
    """
    pa.moveTo(989, 492, duration=0.1)
    pa.click()
    return True

def read_plate():
    """
    start read the plate after choosing a specific protocal.
    """
    pa.moveTo(285, 77, duration=0.1)
    pa.click()


def confirm_loding():
    """
    confirm plate in after clicking read plate
    """
    pa.moveTo(928, 620, duration=0.1)
    pa.click()


def plate_in():
    """
    click the instrument plate in button
    """
    pa.moveTo(909, 529, duration=0.1)
    pa.click()
    time.sleep(0.1)
    # click the carrier in button
    pa.moveTo(1041, 519, duration=0.1)
    pa.click()
    return True




if __name__ == "__main__":
    # init()
    # click_protocal_tab()
    # click_one_glo_protocal()
    # read_plate()
    # confirm_loding()
    pass
