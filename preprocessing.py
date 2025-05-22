import mne
import numpy as np
import pyxdf
import pandas as pd
from autoreject import AutoReject
import matplotlib.pyplot as plt
from mne_icalabel import label_components
from mne.preprocessing import ICA
from scipy import stats

# --- Dizionario di mapping: EEG 1, EEG 2, ... -> nomi canali desiderati ---

xdf_file = r"C:\Users\tonon\Desktop\PYTHON\ses-VR01\sub-P032_ses-VR01_task-Default_run-001_eeg.xdf"
rename_dict = {
    "EEG 1": "AF7",
    "EEG 2": "AF3",
    "EEG 3": "Fp1",
    "EEG 4": "Fp2",
    "EEG 5": "AF4",
    "EEG 6": "AF8",
    "EEG 7": "F7",
    "EEG 8": "F5",
    "EEG 9": "F3",
    "EEG 10": "F1",
    "EEG 11": "F2",
    "EEG 12": "F4",
    "EEG 13": "F6",
    "EEG 14": "F8",
    "EEG 15": "FT7",
    "EEG 16": "FC5",
    "EEG 17": "FC3",
    "EEG 18": "FC1",
    "EEG 19": "FC2",
    "EEG 20": "FC4",
    "EEG 21": "FC6",
    "EEG 22": "FT8",
    "EEG 23": "T3",
    "EEG 24": "C5",
    "EEG 25": "C3",
    "EEG 26": "C1",
    "EEG 27": "C2",
    "EEG 28": "C4",
    "EEG 29": "C6",
    "EEG 30": "T4",
    "EEG 31": "TP7",     
    "EEG 32": "CP5",
    "EEG 33": "CP3",
    "EEG 34": "CP1",
    "EEG 35": "CP2",
    "EEG 36": "CP4",
    "EEG 37": "CP6",
    "EEG 38": "TP8",
    "EEG 39": "T5",
    "EEG 40": "P5",
    "EEG 41": "P3",
    "EEG 42": "P1",
    "EEG 43": "P2",
    "EEG 44": "P4",
    "EEG 45": "P6",
    "EEG 46": "T6",
    "EEG 47": "Fpz",
    "EEG 48": "PO7",
    "EEG 49": "PO3",
    "EEG 50": "O1",
    "EEG 51": "O2",
    "EEG 52": "PO4",
    "EEG 53": "PO8",
    "EEG 54": "Oz",
    "EEG 55": "AFz",
    "EEG 56": "Fz",
    "EEG 57": "FCz",
    "EEG 58": "Cz",
    "EEG 59": "CPz",
    "EEG 60": "Pz",
    "EEG 61": "POz",
}

def common_preprocessing():
    streams, header = pyxdf.load_xdf(xdf_file)
    eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
    if eeg_stream is None:
        raise RuntimeError("No EEG stream found in the XDF file.")


    data = eeg_stream['time_series'].T  # Dati EEG (canali × campioni)

    # Calcola statistiche descrittive
    print(f"Valore massimo: {np.max(data)}")
    print(f"Valore minimo: {np.min(data)}")
    print(f"Valore medio assoluto: {np.mean(np.abs(data))}")
    
    data2 = np.array(eeg_stream['time_series']).T * 1e-6  # convert to Volt
    sfreq2 = float(eeg_stream['info']['nominal_srate'][0])
    n_channels2 = int(eeg_stream['info']['channel_count'][0])
    ch_names2 = [f'EEG {i+1}' for i in range(n_channels2)]
    ch_types2 = ['eeg'] * n_channels2
    info = mne.create_info(ch_names=ch_names2, sfreq=sfreq2, ch_types=ch_types2)
    raw = mne.io.RawArray(data2, info)
    raw.drop_channels("EEG 62")
    
    raw.rename_channels(rename_dict)
    montage2 = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage2, on_missing='ignore')

    raw.resample(256, npad=0)
    raw.filter(0.5, 100)
    raw.notch_filter(freqs=[50], notch_widths=2)
    raw.notch_filter(freqs=[100], notch_widths=2)

    return raw, raw.ch_names

def segment_ica_plot(raw, n_segments=None, n_components=None,
                     method='fastica', random_state=97,
                     segment_duration=15.0, chan_threshold=150.0):
    """
    Per ciascun segmento di `segment_duration` secondi:
      0. Reject channels with |amplitude| > `chan_threshold` µV and interpolate.
      1. Fit ICA su `n_components`.
      2. Calcola la % di varianza spiegata da ogni IC nel segnale RAW
         usando potenza proiettata via mixing matrix.
      3. Seleziona IC che spiegano >3% di varianza.
      4. Per queste, calcola i picchi pos/neg in µV e escludi quelle con |picco|>75 µV.
      5. Etichetta con ICLabel e escludi IC non in ['brain', 'other'].
      6. Applica l’esclusione, ricostruisce e conserva il segmento pulito.
    Alla fine concatena tutto, stampa statistiche e plotta.
    """
    import numpy as np
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    total_dur = raw.times[-1]
    n_segments = int(np.ceil(total_dur / segment_duration))
    if n_components is None:
        n_components = len(raw.ch_names)

    original_var = np.sum(np.var(raw.get_data(), axis=1))

    all_cleaned = []
    seg_count = 0
    total_excluded = 0

    for seg_idx in range(n_segments):
        t0 = seg_idx * segment_duration
        t1 = min(t0 + segment_duration, total_dur)
        if t0 >= total_dur:
            break

        seg = raw.copy().crop(tmin=t0, tmax=t1)
        print(f"\n--- Segmento {seg_idx+1}/{n_segments}: {t0:.1f}s–{t1:.1f}s ---")

        # 0) Reject and interpolate bad channels
        data_seg = seg.get_data()
        # data in V, convert threshold to Volts
        thr_volt = chan_threshold * 1e-6
        peaks = np.max(np.abs(data_seg), axis=1)
        bad_idx = np.where(peaks > thr_volt)[0]
        bad_chs = [seg.ch_names[i] for i in bad_idx]
        if bad_chs:
            print("Canali esclusi (> {0} µV):".format(chan_threshold), bad_chs)
            seg.info['bads'] = bad_chs
            seg.interpolate_bads(reset_bads=True)

        # 1) Fit ICA
        ica = ICA(n_components=n_components, method=method, random_state=random_state)
        ica.fit(seg)

        # 2) Estrai sources e mixing matrix
        S = ica.get_sources(seg).get_data()         # (n_comp, n_times)
        var_s = np.var(S, axis=1)
        mix = ica.mixing_matrix_
        mix_norm2 = np.sum(mix**2, axis=0)

        power_ic = var_s * mix_norm2
        perc_explained = 100 * power_ic / np.sum(power_ic)

        # 3) IC con >3% varianza
        high_var_idx = np.where(perc_explained > 3.0)[0]
        print("IC >3% varianza:", high_var_idx.tolist())

        # 4) Calcola picchi assoluti in µV
        scale = np.sqrt(original_var / np.sum(power_ic))
        S_uv = S * scale * 1e6                       # da V a µV
        peaks_pos = np.max(S_uv, axis=1)
        peaks_neg = np.min(S_uv, axis=1)
        print("Picchi + (µV):", np.round(peaks_pos,1).tolist())
        print("Picchi – (µV):", np.round(peaks_neg,1).tolist())

        to_exclude = [k for k in high_var_idx
                      if (peaks_pos[k] > 75.0) or (peaks_neg[k] < -75.0)]
        print("Escludo per picco>75µV:", to_exclude)

        # 5) ICLabel e ulteriore esclusione
        labels = label_components(seg, ica, method='iclabel')
        label_exclude = [
            i for i, lab in enumerate(labels['labels'])
            if lab not in ['brain', 'other'] and labels['y_pred_proba'][i] > 0.9
        ]
        print("ICLabel - exclude non-brain/other:", label_exclude)

        all_exclude = sorted(set(to_exclude) | set(label_exclude))
        print("Tutte le IC escluse:", all_exclude)

        ica.exclude = all_exclude

        # 6) Applica e conserva pulito
        cleaned = seg.copy()
        ica.apply(cleaned)
        all_cleaned.append(cleaned.get_data())

        total_excluded += len(all_exclude)
        seg_count += 1

    cleaned_data = np.concatenate(all_cleaned, axis=1)
    cleaned_raw = mne.io.RawArray(cleaned_data, raw.info)

    if seg_count:
        avg_exc = total_excluded / seg_count
        kept_var = np.sum(np.var(cleaned_data, axis=1))
        kept_perc = 100 * kept_var / original_var
        print(f"\nMedia IC escluse per segmento: {avg_exc:.2f}")
        print(f"Varianza mantenuta complessiva: {kept_perc:.2f}%")

    cleaned_raw.plot(n_channels=10, scalings='auto',
                     title='EEG pulito dopo ICA + canali bad + filtro',
                     show=True, block=True)
    return cleaned_raw



def load_xdf_with_markers():
    streams, header = pyxdf.load_xdf(xdf_file)
    eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
    markers_stream = next((s for s in streams if s['info']['type'][0] == 'Markers'), None)

    if eeg_stream is None:
        raise RuntimeError("No EEG stream found in the XDF file.")
    if markers_stream is None:
        raise RuntimeError("No Markers stream found in the XDF file.")

    data = np.array(eeg_stream['time_series']).T
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    ch_names = [f'EEG {i+1}' for i in range(data.shape[0])]
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    marker_offset = markers_stream['time_stamps'][0]
    markers = [(timestamp - marker_offset, marker[0]) 
               for timestamp, marker in zip(markers_stream['time_stamps'], markers_stream['time_series'])]
    return markers


def add_markers_to_raw(raw, markers):
    annotations = mne.Annotations(
        onset=[marker[0] for marker in markers],
        duration=[0 for _ in markers],
        description=[marker[1] for marker in markers]
    )
    raw.set_annotations(annotations)


def plot_with_markers(raw):
    raw.plot(n_channels=10, scalings='auto', title='Segnali EEG con Marker', show=True, block=True)


# ---------------------- EXECUTION ---------------------------

raw, chnames = common_preprocessing()
#raw = IC_label(raw, chnames)
#raw.filter(1, 45)
segment_ica_plot(raw, n_segments=10, n_components=48)

#output_csv = r"C:\\Users\\tonon\\Desktop\\PYTHON\\EEG projects\\AutomatedICA\\cleaned_eeg_TEST.csv"
#df_cleaned = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)
#df_cleaned.to_csv(output_csv, index=False)
#print("Dati puliti salvati in CSV:", output_csv)

#markers = load_xdf_with_markers()
#manual_offset = 0
#markers_corrected = [(timestamp + manual_offset, label) for timestamp, label in markers]
#add_markers_to_raw(raw, markers_corrected)
#plot_with_markers(raw, markers_corrected)