from dataclasses import dataclass
import re

@dataclass
class ReadingInfo:
    software_version: str
    experiment_file_path: str
    protocol_file_path: str
    plate_number: str
    date: str
    time: str
    reader_type: str
    reader_serial_number: str
    reading_type: str
    plate_type: str
    eject_plate: str
    read_mode: str
    integration_time: str
    filter_set_emission: str
    mirror: str
    gain: str
    read_speed: str
    delay: str
    extended_dynamic_range: str
    read_height: str
    gain_lum: str
    actual_temperature: str
    results: dict

def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    software_version = re.search(r'Software Version\s+([\d.]+)', content).group(1)
    experiment_file_path = re.search(r'Experiment File Path:\s+(.+)', content).group(1)
    protocol_file_path = re.search(r'Protocol File Path:\s+(.+)', content).group(1)
    plate_number = re.search(r'Plate Number\s+(.+)', content).group(1)
    date = re.search(r'Date\s+(.+)', content).group(1)
    time = re.search(r'Time\s+(.+)', content).group(1)
    reader_type = re.search(r'Reader Type:\s+(.+)', content).group(1)
    reader_serial_number = re.search(r'Reader Serial Number:\s+(.+)', content).group(1)
    reading_type = re.search(r'Reading Type\s+(.+)', content).group(1)
    plate_type = re.search(r'Plate Type\s+(.+)', content).group(1)
    eject_plate = re.search(r'Eject plate on completion\s*(.+)?', content)
    eject_plate = eject_plate.group(1) if eject_plate else ""
    read_mode = re.search(r'Read\s+(.+)', content).group(1)
    integration_time = re.search(r'Integration Time:\s+([\d:.]+)', content).group(1)
    filter_set_emission = re.search(r'Filter Set 1 \(Open\)\s+Emission:\s+(.+)', content).group(1)
    mirror = re.search(r'Mirror:\s+<(.+)>', content).group(1)
    gain = re.search(r'Gain:\s+(.+)', content).group(1)
    read_speed = re.search(r'Read Speed:\s+(.+)', content).group(1)
    delay = re.search(r'Delay:\s+(.+)', content).group(1)
    extended_dynamic_range = re.search(r'Extended Dynamic Range\s*(.+)?', content)
    extended_dynamic_range = extended_dynamic_range.group(1) if extended_dynamic_range else ""
    read_height = re.search(r'Read Height:\s+(.+)', content).group(1)
    gain_lum = re.search(r'Gain\(Lum\)\s+(\d+)', content).group(1)
    actual_temperature = re.search(r'Actual Temperature:\s+(.+)', content).group(1)
    
    results_section = re.search(r'Results\n((?:.+\n)+)', content).group(1)
    results = {}
    for line in results_section.strip().split('\n'):
        if re.match(r'^[A-H]', line):
            key = line[0]
            values = list(map(int, re.findall(r'\d+', line)))
            results[key] = values
    
    reading_info = ReadingInfo(
        software_version, experiment_file_path, protocol_file_path, plate_number, date, time, reader_type, 
        reader_serial_number, reading_type, plate_type, eject_plate, read_mode, integration_time, 
        filter_set_emission, mirror, gain, read_speed, delay, extended_dynamic_range, read_height, gain_lum, 
        actual_temperature, results
    )
    
    return reading_info


if __name__ == '__main__':
    # Usage
    file_path = r'C:\Users\Public\Documents\test\export_api.csv'
    experiment_details = parse_file(file_path)
    print(experiment_details)
