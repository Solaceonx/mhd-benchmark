/* One transport for native video controls and the shared play buttons. */
class SynchronizedVideos {
  constructor(videos, onState = () => {}) {
    this.videos = videos;
    this.onState = onState;
    this.leader = videos[0];
    this.playing = false;
    this.generation = 0;
    this.internalSeeks = new WeakSet();
    for (const video of videos) {
      video.addEventListener('play', () => {
        if (!video.paused && !this.playing) this.play(video);
      });
      video.addEventListener('pause', () => {
        if (video.paused && !video.ended && this.playing) {
          this.leader = video;
          this.pause();
        }
      });
      video.addEventListener('seeking', () => {
        if (!this.internalSeeks.has(video)) {
          this.leader = video;
          this.align(video.currentTime, video);
        }
      });
      video.addEventListener('seeked', () => this.internalSeeks.delete(video));
      video.addEventListener('ratechange', () => {
        for (const peer of videos) {
          if (peer.playbackRate !== video.playbackRate) peer.playbackRate = video.playbackRate;
        }
      });
      video.addEventListener('timeupdate', () => {
        if (this.playing && video === this.leader) this.align(video.currentTime, video, 0.1);
      });
      video.addEventListener('ended', () => {
        if (this.playing) {
          this.pause();
          this.align(video.duration, video, 0.001);
        }
      });
      video.addEventListener('error', () => this.pause('A movie could not load. Reload the page to retry.'));
    }
  }

  align(time, source = null, tolerance = 0.025) {
    for (const video of this.videos) {
      if (video === source) continue;
      const target = Number.isFinite(video.duration) ? Math.min(time, video.duration) : time;
      if (Math.abs(video.currentTime - target) > tolerance) {
        this.internalSeeks.add(video);
        video.currentTime = target;
      }
    }
  }

  async play(source = this.leader) {
    if (this.playing) return;
    this.leader = source;
    this.playing = true;
    const generation = ++this.generation;
    const atEnd = source.ended || (Number.isFinite(source.duration) && source.currentTime >= source.duration - 0.03);
    this.align(atEnd ? 0 : source.currentTime);
    // The calls occur together inside the click/play gesture, including native controls.
    const starts = this.videos.map(video => video.play());
    this.onState(true, 'Linked playback · play, pause, seek and speed apply to all three');
    const results = await Promise.allSettled(starts);
    if (this.generation === generation && results.some(r => r.status === 'rejected')) {
      this.pause('Playback was blocked. Press play again to retry all three movies.');
    }
  }

  pause(message = 'Linked controls · play any movie to resume all three') {
    this.playing = false;
    ++this.generation;
    for (const video of this.videos) if (!video.paused) video.pause();
    this.align(this.leader.currentTime, this.leader, 0.001);
    this.onState(false, message);
  }

  toggle() {
    if (this.playing) this.pause();
    else return this.play();
  }
}

if (typeof window !== 'undefined') window.SynchronizedVideos = SynchronizedVideos;
if (typeof module !== 'undefined') module.exports = SynchronizedVideos;
