from opentrons import protocol_api

# metadata
metadata = {
    'protocolName': "Opentrons_3CR_multi_p20",
    'author': 'Kevin <kevinchang.wang@mail.utoronto.ca',
    'description': 'Opentrons liquid dispensing protocol for 3CR',
    'apiLevel': '2.12'
}

# This protocol will use the Opentrons OT-2 robot to combine the 3 components for the 3 component reaction
# Version: Multi-channel pipette

# Set the component volume here
component_volume = 10

# Change the location of the plates here
linker_head_source_plate_location = '7'
tail_source_plate_location = '10'
destination_plate_location = '11'

# Load component_volume*12 into the specified columns in the linker_head_source_plate
head_component_col = '1'
linker_component_col = '4'

# This protocol requires 3 pipette racks, 1 for each component. Indicate the locations here
tip_rack_1_location = '4'
tip_rack_2_location = '5'
tip_rack_3_location = '6'

# Specifies the transfer of C components (lipid tails) from the source plate well to destination plate columns
C_Component_Dict = {
    'A1': '1',
    'A2': '2',
    'A3': '3',
    'A4': '4',
    'A5': '5',
    'A6': '6',
    'A7': '7',
    'A8': '8',
    'A9': '9',
    'A10': '10',
    'A11': '11',
    'A12': '12'
}


# This protocol will use add 1 lipid_aldehyde to all the wells from the designated column of the linker_head_source_plate
# 8 head groups will be distributed into the designated column of the linker_head_source_plate
# 12 tail groups will be added to the designated column of the plate

# protocol run function
def run(protocol: protocol_api.ProtocolContext):
    # labware
    tail_source_plate = protocol.load_labware('usascientific_12_reservoir_22ml', location=tail_source_plate_location,
                                              label='Tail_Source_Plate')
    linker_head_source_plate = protocol.load_labware('usascientific_12_reservoir_22ml',
                                                     location=linker_head_source_plate_location,
                                                     label='Linker_Source_Plate')

    destination_plate = protocol.load_labware('vwr_96_pcr_wellplate_200ul', location='11',
                                              label='Destination_Plate')
    tiprack_1 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_1_location, label='Tip_Rack_1')
    tiprack_2 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_2_location, label='Tip_Rack_2')
    tiprack_3 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_3_location, label='Tip_Rack_3')
    # pipette
    p20_multi_pipette = protocol.load_instrument('p20_multi_gen2', mount='left',
                                                  tip_racks=[tiprack_1, tiprack_2, tiprack_3])
    # distribute head components to an empty plate
    p20_multi_pipette.pick_up_tip()
    p20_multi_pipette.transfer(component_volume, linker_head_source_plate.columns_by_name()[head_component_col],
                                destination_plate.wells(), new_tip='never')
    p20_multi_pipette.drop_tip()

    # distribute lipid_aldehyde components to an empty plate
    p20_multi_pipette.transfer(component_volume, linker_head_source_plate.columns_by_name()[linker_component_col],
                                destination_plate.wells(),
                                new_tip='always')

    # distribute C components to the designated columns

    for component in C_Component_Dict:
        p20_multi_pipette.transfer(component_volume, tail_source_plate.wells_by_name()[component],
                                    destination_plate.columns_by_name()[C_Component_Dict[component]], new_tip='always')
