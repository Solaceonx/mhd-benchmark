// Exercise asynchronous media events without requiring a DOM or browser package.
const test = require('node:test');
const assert = require('node:assert/strict');
const SynchronizedVideos = require('../progress/sync-videos.js');
const flush = () => new Promise(resolve => setImmediate(resolve));

class Video extends EventTarget {
  constructor() { super(); this.paused=true; this.ended=false; this.duration=8.333333; this.time=0; this.rate=1; this.rejectPlay=false; }
  fire(event) { queueMicrotask(()=>this.dispatchEvent(new Event(event))); }
  get currentTime() { return this.time; }
  set currentTime(value) { this.time=value; this.ended=false; this.fire('seeking'); this.fire('seeked'); }
  get playbackRate() { return this.rate; }
  set playbackRate(value) { this.rate=value; this.fire('ratechange'); }
  play() {
    if(this.rejectPlay)return Promise.reject(new Error('blocked'));
    if(this.paused){this.paused=false;this.ended=false;this.fire('play');}
    return Promise.resolve();
  }
  pause() { if(!this.paused){this.paused=true;this.fire('pause');} }
}

test('native play on each player starts all; native pause stops all',async()=>{
  for(let source=0;source<3;source++){
    const videos=Array.from({length:3},()=>new Video());new SynchronizedVideos(videos);
    await videos[source].play();await flush();assert.ok(videos.every(v=>!v.paused));
    videos[(source+1)%3].pause();await flush();assert.ok(videos.every(v=>v.paused));
  }
});

test('seek, rate and playback drift propagate without a feedback loop',async()=>{
  const videos=Array.from({length:3},()=>new Video());const transport=new SynchronizedVideos(videos);
  await transport.play();await flush();
  videos[2].currentTime=3.5;await flush();assert.ok(videos.every(v=>v.currentTime===3.5));
  videos[1].playbackRate=1.5;await flush();assert.ok(videos.every(v=>v.playbackRate===1.5));
  videos[2].time=4;videos[2].fire('timeupdate');await flush();assert.ok(videos.every(v=>v.currentTime===4));
});

test('pause/resume keeps the time, and end/replay returns all to zero',async()=>{
  const videos=Array.from({length:3},()=>new Video());const transport=new SynchronizedVideos(videos);
  videos[0].currentTime=2;await flush();await transport.toggle();await flush();transport.toggle();await flush();
  await transport.toggle();await flush();assert.ok(videos.every(v=>v.currentTime===2&&!v.paused));
  videos[0].time=videos[0].duration;videos[0].ended=true;videos[0].paused=true;videos[0].fire('ended');await flush();
  assert.ok(videos.every(v=>v.paused));await transport.toggle();await flush();
  assert.ok(videos.every(v=>v.currentTime===0&&!v.paused));
});

test('one rejected play stops the group and a later retry works',async()=>{
  const videos=Array.from({length:3},()=>new Video());let message='';
  const transport=new SynchronizedVideos(videos,(_,text)=>message=text);
  videos[2].rejectPlay=true;await transport.play();await flush();
  assert.ok(videos.every(v=>v.paused));assert.match(message,/blocked/);
  videos[2].rejectPlay=false;await transport.play();await flush();assert.ok(videos.every(v=>!v.paused));
});
