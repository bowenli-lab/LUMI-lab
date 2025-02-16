from opentrons import protocol_api

# This code has been simply modified for round bottom plates by increasing the dispense height from 10 to 12 mm. However, the destination plate is still listed as the flat bottom plate.

# metadata
metadata = {
    'protocolName': "Add OneGlo - Round Bottom Plates",
    'author': 'Kevin <kevinchang.wang@mail.utoronto.ca>',
    'description': 'Opentrons liquid dispensing protocol to add OneGlo to round bottom plates following transfection',
    'apiLevel': '2.12'
}

# amounts
oneglo_amount = 40

# locations
tip_rack_location = 10
oneglo_locatiion = 11

cell_plate_1_location = 1
cell_plate_2_location = 2
cell_plate_3_location = 3
cell_plate_4_location = 4
cell_plate_5_location = 5
cell_plate_6_location = 6
cell_plate_7_location = 7
cell_plate_8_location = 8
cell_plate_9_location = 9


# run
def run(protocol: protocol_api.ProtocolContext):
    # load tip racks
    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', location=tip_rack_location, label='Tip_Rack')

    # load pipette
    p300_multi = protocol.load_instrument('p300_multi_gen2', mount='right', tip_racks=[tiprack])

    # load oneglo plate

    oneglo_plate = protocol.load_labware('usascientific_12_reservoir_22ml', location=oneglo_locatiion, label='One-glo')

    # load cell plates
    cell_plate_1 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_1_location,
                                         label='Cell_Plate_1')
    cell_plate_2 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_2_location,
                                         label='Cell_Plate_2')
    cell_plate_3 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_3_location,
                                         label='Cell_Plate_3')
    cell_plate_4 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_4_location,
                                         label='Cell_Plate_4')
    cell_plate_5 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_5_location,
                                         label='Cell_Plate_5')
    cell_plate_6 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_6_location,
                                         label='Cell_Plate_6')
    cell_plate_7 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_7_location,
                                         label='Cell_Plate_7')
    cell_plate_8 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_8_location,
                                         label='Cell_Plate_8')
    cell_plate_9 = protocol.load_labware('thermoscientific_96_wellplate_200ul', location=cell_plate_9_location,
                                         label='Cell_Plate_9')

    # add oneglo
    p300_multi.well_bottom_clearance.dispense = 12
    p300_multi.pick_up_tip()
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['1'], cell_plate_1.wells(), new_tip='never', blow_out=True, blowout_location='destination well')
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['2'], cell_plate_2.wells(), new_tip='never', blow_out=True, blowout_location='destination well')
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['3'], cell_plate_3.wells(), new_tip='never', blow_out=True, blowout_location='destination well')
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['4'], cell_plate_4.wells(), new_tip='never', blow_out=True, blowout_location='destination well')
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['5'], cell_plate_5.wells(), new_tip='never', blow_out=True, blowout_location='destination well')
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['6'], cell_plate_6.wells(), new_tip='never', blow_out=True, blowout_location='destination well')
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['7'], cell_plate_7.wells(), new_tip='never', blow_out=True, blowout_location='destination well')
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['8'], cell_plate_8.wells(), new_tip='never', blow_out=True, blowout_location='destination well')
    p300_multi.transfer(oneglo_amount, oneglo_plate.columns_by_name()['9'], cell_plate_9.wells(), new_tip='never', blow_out=True, blowout_location='destination well')

