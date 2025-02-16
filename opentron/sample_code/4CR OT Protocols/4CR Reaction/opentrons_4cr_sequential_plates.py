from opentrons import protocol_api

# metadata
metadata = {
    'protocolName': "Opentrons_4CR_sequential_plates",
    'author': 'Kevin <kevinchang.wang@mail.utoronto.ca',
    'description': 'Opentrons liquid dispensing protocol for 4CR, sequential plate filling',
    'apiLevel': '2.12'
}

# This protocol will use the Opentrons OT-2 robot to combine the 4 components for the 4 component reaction
# Destination plates are filled sequentially instead of simultanously
# Version: Multi-channel pipette

# Set the component volume here
component_volume = 25

# Change the location of the plates here
source_plate_location = '5'
destination_plate_1_location = '1'
destination_plate_2_location = '3'

# Change the type of plates here
source_plate_type = 'thermoscientificnunc_96_wellplate_1300ul'
destination_plate_type = 'servicebio_96_wellplate_200ul'

# This protocol requires 8 pipette racks. Indicate the locations here
tip_rack_1_location = '10'
tip_rack_2_location = '11'
tip_rack_3_location = '7'
tip_rack_4_location = '8'
tip_rack_5_location = '9'
tip_rack_6_location = '4'
tip_rack_7_location = '6'
tip_rack_8_location = '2'

# Set the tip rack type here
tip_rack_type = 'opentrons_96_tiprack_300ul'

# Set the pipette parameters here
pipette_type = 'p300_multi_gen2'
pipette_mount = 'left'

# Indicate the location of each component here

# Linker locations
linker_dict_1 = {
    '1': '1',
    '2': '3',
    '3': '5',
    '4': '7',
    '5': '9',
    '6': '11'
}
linker_dict_2 = {
    '1': '2',
    '2': '4',
    '3': '6',
    '4': '8',
    '5': '10',
    '6': '12'
}
# Tail A locations
tail_A1_A3_location = '7'
tail_A1_A3_destination = ['1', '3', '5', '7', '9', '11']
tail_A2_A4_location = '8'
tail_A2_A4_destination = ['2', '4', '6', '8', '10', '12']

# Tail B locations
tail_B_location = '9'

# Headgroup locations
headgroup_1_location = '10'
headgroup_2_location = '11'


def run(protocol: protocol_api.ProtocolContext):
    # load labware
    # load plates
    source_plate = protocol.load_labware(source_plate_type, location=source_plate_location, label="source_plate")
    destination_plate_1 = protocol.load_labware(destination_plate_type, location=destination_plate_1_location,
                                                label='destination_plate_1')
    destination_plate_2 = protocol.load_labware(destination_plate_type, location=destination_plate_2_location,
                                                label='destination_plate_2')

    # load tipracks
    tip_rack_1 = protocol.load_labware(tip_rack_type, location=tip_rack_1_location, label='tip_rack_1')
    tip_rack_2 = protocol.load_labware(tip_rack_type, location=tip_rack_2_location, label='tip_rack_2')
    tip_rack_3 = protocol.load_labware(tip_rack_type, location=tip_rack_3_location, label='tip_rack_3')
    tip_rack_4 = protocol.load_labware(tip_rack_type, location=tip_rack_4_location, label='tip_rack_4')
    tip_rack_5 = protocol.load_labware(tip_rack_type, location=tip_rack_5_location, label='tip_rack_5')
    tip_rack_6 = protocol.load_labware(tip_rack_type, location=tip_rack_6_location, label='tip_rack_6')
    tip_rack_7 = protocol.load_labware(tip_rack_type, location=tip_rack_7_location, label='tip_rack_7')
    tip_rack_8 = protocol.load_labware(tip_rack_type, location=tip_rack_8_location, label='tip_rack_8')

    # load pipette
    pipette = protocol.load_instrument(pipette_type, mount=pipette_mount,
                                       tip_racks=[tip_rack_1, tip_rack_2, tip_rack_3, tip_rack_4, tip_rack_5,
                                                  tip_rack_6, tip_rack_7, tip_rack_8])

    # distribute components

    # DISTRIBUTE COMPONENTS TO DESTINATION PLATE 1

    # distribute headgroups to destination plate 1
    pipette.pick_up_tip()
    pipette.transfer(component_volume, source_plate.columns_by_name()[headgroup_1_location],
                     destination_plate_1.wells(), new_tip='never')
    pipette.drop_tip()

    # distribute linkers to destination plate 1
    for component in linker_dict_1:
        pipette.transfer(component_volume, source_plate.columns_by_name()[component],
                         destination_plate_1.columns_by_name()[linker_dict_1[component]], new_tip='always')
    for component in linker_dict_2:
        pipette.transfer(component_volume, source_plate.columns_by_name()[component],
                         destination_plate_1.columns_by_name()[linker_dict_2[component]], new_tip='always')

    # distribute tail A to destination plate 1
    for component in tail_A1_A3_destination:
        pipette.transfer(component_volume, source_plate.columns_by_name()[tail_A1_A3_location],
                         destination_plate_1.columns_by_name()[component], new_tip='always')
    for component in tail_A2_A4_destination:
        pipette.transfer(component_volume, source_plate.columns_by_name()[tail_A2_A4_location],
                         destination_plate_1.columns_by_name()[component], new_tip='always')

    # distribute tail B to destination plate 1
    pipette.transfer(component_volume, source_plate.columns_by_name()[tail_B_location], destination_plate_1.wells(),
                     new_tip='always')

    # DISTRIBUTE COMPONENTS TO DESTINATION PLATE 2

    # distribute headgroups to destination plate 2
    pipette.pick_up_tip()
    pipette.transfer(component_volume, source_plate.columns_by_name()[headgroup_2_location],
                     destination_plate_2.wells(), new_tip='never')
    pipette.drop_tip()
    # distribute linkers to destination plate 2
    for component in linker_dict_1:
        pipette.transfer(component_volume, source_plate.columns_by_name()[component],
                         destination_plate_2.columns_by_name()[linker_dict_1[component]], new_tip='always')
    for component in linker_dict_2:
        pipette.transfer(component_volume, source_plate.columns_by_name()[component],
                         destination_plate_2.columns_by_name()[linker_dict_2[component]], new_tip='always')

    # distribute tail A to destination plate 2
    for component in tail_A1_A3_destination:
        pipette.transfer(component_volume, source_plate.columns_by_name()[tail_A1_A3_location],
                         destination_plate_2.columns_by_name()[component], new_tip='always')
    for component in tail_A2_A4_destination:
        pipette.transfer(component_volume, source_plate.columns_by_name()[tail_A2_A4_location],
                         destination_plate_2.columns_by_name()[component], new_tip='always')
    # distribute tail B to destination plate 2
    pipette.transfer(component_volume, source_plate.columns_by_name()[tail_B_location], destination_plate_2.wells(),
                     new_tip='always')
