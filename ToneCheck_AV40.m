% Parameters
amp = 1;              % Amplitude
freq = 4000;         % Frequency of tone (16 kHz)
Fs = 96000;           % Sampling rate (48 kHz)
dur = 0.1;          % Duration in seconds

% Generate the time vector
tVec = 0:1/Fs:dur - 1/Fs;

% Generate the 16 kHz sine wave
tone = amp * sin(2 * pi * freq * tVec);

% Plot the waveform in time domain
figure;
subplot(3,1,1);
plot(tVec, tone);
xlabel('Time (s)');
ylabel('Amplitude');
title('4 kHz Tone at 48 kHz Sampling Rate');
xlim([0 5/freq]); % Display only a few cycles for clarity

% Parameters
amp = 1;              % Amplitude
freq = 12000;         % Frequency of tone (16 kHz)
Fs = 96000;           % Sampling rate (48 kHz)
dur = 0.1;          % Duration in seconds

% Generate the time vector
tVec = 0:1/Fs:dur - 1/Fs;

% Generate the 16 kHz sine wave
tone = amp * sin(2 * pi * freq * tVec);

% Plot the waveform in time domain
hold on;
subplot(3,1,2);
plot(tVec, tone);
xlabel('Time (s)');
ylabel('Amplitude');
title('12 kHz Tone at 48 kHz Sampling Rate');
xlim([0 5/freq]); % Display only a few cycles for clarity

% Parameters
amp = 1;              % Amplitude
freq = 16000;         % Frequency of tone (16 kHz)
Fs = 48000;           % Sampling rate (48 kHz)
dur = 0.1;          % Duration in seconds

% Generate the time vector
tVec = 0:1/Fs:dur - 1/Fs;

% Generate the 16 kHz sine wave
tone = amp * sin(2 * pi * freq * tVec);

% Plot the waveform in time domain
hold on;
subplot(3,1,3);
plot(tVec, tone);
xlabel('Time (s)');
ylabel('Amplitude');
title('16 kHz Tone at 48 kHz Sampling Rate');
xlim([0 5/freq]); % Display only a few cycles for clarity

% % Calculate and plot the frequency spectrum
N = length(tone);
fVec = (0:N-1)*(Fs/N); % Frequency vector for plotting
tone_fft = abs(fft(tone)); % FFT to get the frequency domain representation
subplot(2,1,2);
plot(fVec(1:N/2), tone_fft(1:N/2)); % Plot only the positive frequencies
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Spectrum of 16 kHz Tone');
xlim([0 Fs/2]); % Limit x-axis to Nyquist frequency
