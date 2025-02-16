from opentrons import protocol_api

# metadata
metadata = {
    'protocolName': "4CR LNP Formation - no centrifuge",
    'author': 'Kevin <kevinchang.wang@mail.utoronto.ca',
    'description': 'Opentrons liquid dispensing protocol for 4cr lipid nanoparticle formation, sequential',
    'apiLevel': '2.12'
}

# Set to True to include optional pause
centrifuge_pause = False
between_plate_pause = False

# Set the component amounts here
etoh_mm_amount = 2.25
il_amount = 3.86
aq_mm_amount = 18.33

# Change the plate information here.
# Source plates
aq_mm_plate_location = '10'
aq_mm_well = '1'
aq_mm_plate_type = 'usascientific_12_reservoir_22ml'

etoh_mm_plate_location = '7'
etoh_mm_well = '1'
etoh_mm_plate_type = 'servicebio_96_wellplate_200ul'

h1_il_plate_location = '4'
h1_il_plate_type = 'servicebio_96_wellplate_200ul'

h2_il_plate_location = '5'
h2_il_plate_type = 'servicebio_96_wellplate_200ul'
# Destination plates
h1_lnp_plate_location = '1'
h1_lnp_plate_type = 'servicebio_96_wellplate_200ul'

h2_lnp_plate_location = '2'
h2_lnp_plate_type = 'servicebio_96_wellplate_200ul'

# This protocol requires 5 tip racks. Change the tip rack information here.

tip_rack_type = 'opentrons_96_tiprack_20ul'
tip_rack_1_location = '11'
tip_rack_2_location = '8'
tip_rack_3_location = '9'
tip_rack_4_location = '6'
tip_rack_5_location = '3'

# Change pipette information here.
pipette_type = 'p20_multi_gen2'
pipette_mount = 'left'


def run(protocol: protocol_api.ProtocolContext):
    # LOAD LABWARE
    # Source plates
    aq_mm_source_plate = protocol.load_labware(aq_mm_plate_type, location=aq_mm_plate_location,
                                               label='aqueous_mm_source_plate')
    etoh_mm_source_plate = protocol.load_labware(etoh_mm_plate_type, location=etoh_mm_plate_location,
                                                 label='etoh_mm_source_plate')
    h1_il_source_plate = protocol.load_labware(h1_il_plate_type, location=h1_il_plate_location,
                                               label='h1_ionizable_lipid_source_plate')
    h2_il_source_plate = protocol.load_labware(h2_il_plate_type, location=h2_il_plate_location,
                                               label='h2_ionizable_lipid_source_plate')

    # Destination plates
    h1_lnp_dest_plate = protocol.load_labware(h1_lnp_plate_type, location=h1_lnp_plate_location,
                                              label='h1_lnp_dest_plate')
    h2_lnp_dest_plate = protocol.load_labware(h2_lnp_plate_type, location=h2_lnp_plate_location,
                                              label='h2_lnp_dest_plate')

    # Tipracks
    tip_rack_1 = protocol.load_labware(tip_rack_type, location=tip_rack_1_location, label='tip_rack_1')
    tip_rack_2 = protocol.load_labware(tip_rack_type, location=tip_rack_2_location, label='tip_rack_2')
    tip_rack_3 = protocol.load_labware(tip_rack_type, location=tip_rack_3_location, label='tip_rack_3')
    tip_rack_4 = protocol.load_labware(tip_rack_type, location=tip_rack_4_location, label='tip_rack_4')
    tip_rack_5 = protocol.load_labware(tip_rack_type, location=tip_rack_5_location, label='tip_rack_5')

    # Pipette
    pipette = protocol.load_instrument(pipette_type, mount=pipette_mount,
                                       tip_racks=[tip_rack_1, tip_rack_2, tip_rack_3, tip_rack_4, tip_rack_5])

    # TRANSFER COMPONENTS
    # Transfer EtOH MM to H1 LNP destination plate
    pipette.pick_up_tip()
    pipette.transfer(etoh_mm_amount, etoh_mm_source_plate.columns_by_name()[etoh_mm_well], h1_lnp_dest_plate.wells(),
                     new_tip='never')
    pipette.drop_tip()

    # Transfer H1 IL to H1 LNP destination plate
    pipette.transfer(il_amount, h1_il_source_plate.wells(), h1_lnp_dest_plate.wells(), new_tip='always')

    # Optional pause
    if centrifuge_pause:
        protocol.pause('Centrifuge time')

    # Transfer aqueous MM to H1 LNP destination plate, mix
    pipette.transfer(aq_mm_amount, aq_mm_source_plate.columns_by_name()[aq_mm_well], h1_lnp_dest_plate.wells(),
                     new_tip='always', mix_after=(3, aq_mm_amount))

    # Optional pause
    if between_plate_pause:
        protocol.pause('Next plate')

    # Transfer EtOH MM to H2 LNP destination plate
    pipette.pick_up_tip()
    pipette.transfer(etoh_mm_amount, etoh_mm_source_plate.columns_by_name()[etoh_mm_well], h2_lnp_dest_plate.wells(),
                     new_tip='never')
    pipette.drop_tip()

    # Transfer H2 IL to H2 LNP destination plate
    pipette.transfer(il_amount, h2_il_source_plate.wells(), h2_lnp_dest_plate.wells(), new_tip='always')

    # Optional pause
    if centrifuge_pause:
        protocol.pause('Centrifuge time')

    # Transfer aqueous MM to H2 LNP destination plate, mix
    pipette.transfer(aq_mm_amount, aq_mm_source_plate.columns_by_name()[aq_mm_well], h2_lnp_dest_plate.wells(),
                     new_tip='always', mix_after=(3, aq_mm_amount))



