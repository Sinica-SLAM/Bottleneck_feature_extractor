import numpy as np

def extfrm(data, threshold=-20):
	"""Extract frame over the power threshold
	Parameters
	----------
	data: array, shape (`T`, `dim`)
		Array of input data
	threshold: scala, optional
		Scala of power threshold [dB]
		Default set to -20
	Returns
	-------
	valid_index: array, shape (`T_ext`)
		Remaining index after extracting frame
		`T_ext` <= `T`
	"""
	npow = spc2npow(data)
	valid_index = np.where(npow > threshold)
	if isinstance(valid_index, tuple):
		return valid_index[0]
	else:
		return valid_index

def spc2npow(spectrogram, eps=1e-12):
	"""Calculate normalized_melcd power sequence from spectrogram
	Parameters
	----------
	spectrogram : array, shape (T, `fftlen / 2 + 1`)
		Array of specturm envelope
	Return
	------
	npow : array, shape (`T`, `1`)
		Normalized power sequence
	"""
	# frame based processing
	npow = np.apply_along_axis(_spvec2pow, 1, spectrogram)
	meanpow = np.mean(npow) + eps
	npow = 10.0 * np.log10(npow / meanpow)

	return npow

def _spvec2pow(specvec):
	"""Convert vector of spectrum envelope into normalized power
	Parameters
	----------
	specvec : vector, shape (`fftlen / 2 + 1`)
		Cector of specturm envelope

	Return
	------
	power : scala,
		Normalized power
	"""
	# set FFT length
	fftl2 = len(specvec) - 1
	fftl = fftl2 * 2
	# specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
	power = specvec[0] + specvec[fftl2]
	for k in range(1, fftl2):
		power += 2.0 * specvec[k]
	power /= fftl
	return power