import sound from './handler-sound.js'

export default (() => {
    let used = false
    const E = {Sensor: class{}, use() {
        if (used) return
        console.log(sound) // TODO: This can't be done immediately... Should the assignment be done on first use, maybe?
        E.sound = sound
        used = true
    }}
    return E
})()