# Bertrino: Neutrino Direction Prediction using BERT-style Transformers

## Goal of the Competition

The goal of this competition is to predict a neutrino particle's direction using data from the "IceCube" detector, which observes the cosmos from deep within the South Pole ice. Your work could help scientists better understand exploding stars, gamma-ray bursts, and cataclysmic phenomena involving black holes, neutron stars, and the fundamental properties of the neutrino itself.

## Context

Neutrinos are one of the most abundant particles in the universe. They are similar to electrons but are nearly massless and electrically neutral, making them difficult to detect. Scientists must estimate the direction of neutrino events to gather enough information to probe the most violent astrophysical sources. If algorithms could be made considerably faster and more accurate, it would allow for more neutrino events to be analyzed, possibly even in real-time, and dramatically increase the chance to identify cosmic neutrino sources. Rapid detection could enable networks of telescopes worldwide to search for more transient phenomena.

Researchers have developed multiple approaches over the past ten years to reconstruct neutrino events. However, problems arise as existing solutions are far from perfect. They're either fast but inaccurate or more accurate at the price of huge computational costs.

The IceCube Neutrino Observatory is the first detector of its kind, encompassing a cubic kilometer of ice and designed to search for the nearly massless neutrinos. An international group of scientists is responsible for the scientific research that makes up the IceCube Collaboration.

By making the process faster and more precise, you'll help improve the reconstruction of neutrinos. As a result, we could gain a clearer image of our universe.

## Dataset Description

The goal of this competition is to identify which direction neutrinos detected by the IceCube neutrino observatory came from. When detection events can be localized quickly enough, traditional telescopes are recruited to investigate short-lived neutrino sources such as supernovae or gamma ray bursts. Because the sky is huge better localization will not only associate neutrinos with sources but also to help partner observatories limit their search space. With an average of three thousand events per second to process, it's difficult to keep up with the stream of data using traditional methods. Your challenge in this competition is to quickly and accurately process a large number of events.

This competition uses a hidden test set. When your submitted notebook is scored the actual test data (including a full length sample submission) will be made available to your notebook. Expect to see roughly one million events in the hidden test set, split between multiple batches.

### Files

**[train/test]_meta.parquet**
- batch_id (int): the ID of the batch the event was placed into.
- event_id (int): the event ID.
- [first/last]_pulse_index (int): index of the first/last row in the features dataframe belonging to this event.
- [azimuth/zenith] (float32): the [azimuth/zenith] angle in radians of the neutrino. A value between 0 and 2*pi for the azimuth and 0 and pi for zenith. The target columns. Not provided for the test set. The direction vector represented by zenith and azimuth points to where the neutrino came from.

*NB: Other quantities regarding the event, such as the interaction point in x, y, z (vertex position), the neutrino energy, or the interaction type and kinematics are not included in the dataset.*

**[train/test]/batch_[n].parquet**
Each batch contains tens of thousands of events. Each event may contain thousands of pulses, each of which is the digitized output from a photomultiplier tube and occupies one row.
- event_id (int): the event ID. Saved as the index column in parquet.
- time (int): the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.
- sensor_id (int): the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.
- charge (float32): An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.). A physical photon does not exactly result in a measurement of 1 p.e. but rather can take values spread around 1 p.e. As an example, a pulse with charge 2.7 p.e. could quite likely be the result of two or three photons hitting the photomultiplier tube around the same time. This data has float16 precision but is stored as float32 due to limitations of the version of pyarrow the data was prepared with.
- auxiliary (bool): If True, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise. If False, then this pulse was contributed to the trigger decision and the pulse was fully digitized.

**sample_submission.parquet**
An example submission with the correct columns and properly ordered event IDs. The sample submission is provided in the parquet format so it can be read quickly but your final submission must be a csv.

**sensor_geometry.csv**
The x, y, and z positions for each of the 5160 IceCube sensors. The row index corresponds to the sensor_idx feature of pulses. The x, y, and z coordinates are in units of meters, with the origin at the center of the IceCube detector. The coordinate system is right-handed, and the z-axis points upwards when standing at the South Pole.

## **Bertrino**: BERT-style Transformers for Neutrino Direction Prediction

Bertrino is an attempt to solve this challenge using BERT-style transformers and pretraining. By leveraging the powerful representation learning capabilities of transformers, we aim to create a model that can quickly and accurately predict the direction of neutrino events, ultimately helping scientists gain a better understanding of our universe.
