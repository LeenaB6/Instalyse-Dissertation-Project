const express = require('express')
const app = express()

app.get('/getPost', async (req, res) => { //Communicates and queries the instagram graph api to get the user's profile and post information
    console.log("req = " + req.query.targetUser)
    const version = "v3.2"
    const myID = "" //INSERT ID for your Instagram account
    
    const username = req.query.targetUser.trim()
 
    const access_token = "" // INSERT your access token here
    const resFields = ["followers_count", "biography", "username", "name", "profile_picture_url", "media_count", "media{id, caption, media_type, like_count, comment_count, timestamp, username, media_url, children{media_url}}"]
    const response = await fetch(`https://graph.facebook.com/${version}/${myID}?fields=business_discovery.username(${username}){${resFields.join(',')}}&access_token=${access_token}`)
    const json = await response.json()
    res.send(json)
    console.log(json)
})

app.get('/getSenti', async (req, res) => { //queries the flask server to run the sentiment analysis model
    const caption = req.query['caption']
    try {
        const backendURL = `http://127.0.0.1:5000/senti?caption=${caption}`; //Passes caption to AI through flask application
        const responseArr = await fetch(backendURL);
        const resJSON = await responseArr.json()
        console.log("resJSON = " +resJSON)
        res.send(resJSON)
    } catch (error) {
        res.send("sentiAnalysis experienced an error - " + error)
    }
})

app.get('/getTopic', async (req, res) => { //queries the flask server to run the topic identification model
    const caption = req.query['caption']
    try {
        const backendURL = `http://127.0.0.1:5001/topic?caption=${caption}`; //Passes caption to AI through flask application
        const resTopic = await fetch(backendURL);
        const resJSONTopic = await resTopic.json()
        console.log("res = " + resJSONTopic)
        res.send(resJSONTopic)
    } catch (error) {
        res.send("topicIdentification experienced an error - " + error)
    }
})

app.listen(3001) 