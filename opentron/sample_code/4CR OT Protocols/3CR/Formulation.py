from opentrons import protocol_api

# metadata
metadata = {
    'protocolName': "3CR lipids to nanoparticle transfection",
    'author': 'Kevin <kevinchang.wang@mail.utoronto.ca',
    'description': 'Opentrons liquid dispensing protocol for 3cr lipids to nanoparticle transfection',
    'apiLevel': '2.12'
}

ionizable_lipid_amount = 5
lipid_master_mix_amount = 3.5
aqueous_phase_amount = 25.4
nanoparticle_amount = 3

tip_rack_1_location = '10'
tip_rack_2_location = '7'
tip_rack_3_location = '4'
tip_rack_4_location = '1'

master_mix_source_plate_location = '11'
master_mix_col = '1'
ionizable_lipid_source_plate_location = '2'
nanoparticle_plate_location = '5'
mrna_plate_location = '8'
mrna_col = '1'
cell_plate_location = '9'


def run(protocol: protocol_api.ProtocolContext):
    # Load Labware
    # Load Plates
    master_mix_source_plate = protocol.load_labware('vwr_96_pcr_wellplate_200ul',
                                                    location=master_mix_source_plate_location,
                                                    label='Master_Mix_Source_Plate')
    ionizable_lipid_source_plate = protocol.load_labware('vwr_96_pcr_wellplate_200ul',
                                                         location=ionizable_lipid_source_plate_location,
                                                         label='Ionizable_Lipid_Plate')
    nanoparticle_plate = protocol.load_labware('vwr_96_pcr_wellplate_200ul', location=nanoparticle_plate_location,
                                               label='Nanoparticle_Destination_Plate')
    mrna_source_plate = protocol.load_labware('usascientific_12_reservoir_22ml', location=mrna_plate_location,
                                              label='mRNA_Plate')

    # Load Tipracks
    tiprack_1 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_1_location, label='Tip_Rack_1')
    tiprack_2 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_2_location, label='Tip_Rack_2')
    tiprack_3 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_3_location, label='Tip_Rack_2')
    tiprack_4 = protocol.load_labware('opentrons_96_tiprack_300ul', location=tip_rack_4_location, label='Tip_Rack_4')
    # Load Pipettes
    p20_multi_pipette = protocol.load_instrument('p20_multi_gen2', mount='left',
                                                 tip_racks=[tiprack_1, tiprack_2, tiprack_3])
    p300_multi_pipette = protocol.load_instrument('p300_multi_gen2', mount='right', tip_racks=[tiprack_4])

    # Add master mix to empty PCR plate
    p20_multi_pipette.pick_up_tip()
    p20_multi_pipette.transfer(lipid_master_mix_amount, master_mix_source_plate.columns_by_name()[master_mix_col],
                               nanoparticle_plate.wells(), new_tip='never')
    p20_multi_pipette.drop_tip()

    # Add ionizable lipid to PCR plate and mix
    p20_multi_pipette.transfer(ionizable_lipid_amount, ionizable_lipid_source_plate.wells(),
                               nanoparticle_plate.wells(), new_tip='always')

    # Add aqueous phase to the organic phase and mix
    p300_multi_pipette.transfer(aqueous_phase_amount, mrna_source_plate.columns_by_name()[mrna_col],
                                nanoparticle_plate.wells(), new_tip='always', mix_after=(5, aqueous_phase_amount))
