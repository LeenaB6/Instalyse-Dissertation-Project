import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import Analysis from "./components/Analysis";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { TagCloud } from "react-tagcloud";
//import Facebook from "./components/Facebook"; //only needed for facebook log in implementation

function App() {
  //Functional component for App

  const [postURLs, setURLs] = useState([]); //Defines the global variables needed for the application
  const [profileTopics, setProfileTopics] = useState([]);
  const [profilePicURL, setProPicURL] = useState();
  const [userFollowers, setFollowers] = useState();
  const [userBiography, setBio] = useState();
  const [userUsername, setUsername] = useState();
  const [usersName, setName] = useState();
  const [profilePerc, setProfilePerc] = useState();
  const [wordCloudData, setWordCloudData] = useState([]);
  const [error, setErr] = useState(false);


  function setWordCloud(input) {  //Sets the word cloud of topics for a profile as a whole
    let topicCounter = {}; //array of the amount of times a topic is determined for a profile
    for (let i = 0; i < input.length; i++) { //Loops through the number of topics there are
      const key = input[i]; //uses the topic as the key
      if (!topicCounter[input[i]]) topicCounter[input[i]] = 1; //if count doesnt exist for that topic then initialise it as one
      else topicCounter[input[i]] += 1; //else, add 1 to the existing count
    }
    let topicsArray = [] //array of topics
    console.log("topic count = " + JSON.stringify(topicCounter))
    for (const [key, value] of Object.entries(topicCounter)) { //loops through the topics and their count values
      topicsArray.push({value: key, count: value}) //adds them one by one to the topic array
    }
    console.log("topic arr = " + JSON.stringify(topicsArray))
    return topicsArray; //returns the full array of topics and their corresponding counts in the format that the tag-cloud takes as its input
  }

  function Search() { //search function to handle searching for a user
    const [targetUser, setTargetUser] = useState(""); 
    //return handles having a button and actions the handleJSON function on button click - also handles validation if the user enters invalid username
    return (
      <div className="searchContainer">
        <input
          className="searchInput"
          type="text"
          name="targetUser"
          onChange={(e) => {setTargetUser(e.target.value); setErr(false)}}
        />
        <button
          className="searchButton"
          title="Search for User"
          onClick={handleJSON}
        >
          Search for User
        </button>
        <div className="errorTitle" style={{opacity: error ? 1 : 0 }}>Invalid username, please try again!</div>
      </div>
    );

    //Function gets the JSON info of the profile and post from instagram graph api
    async function handleJSON() {
      const res = await fetch(`/getPost?targetUser=${targetUser}`);
      const json = await res.json();
      if (json.error) return setErr(true)
      console.log("json obj " + JSON.stringify(json));
      const media = await json.business_discovery.media.data;
      const getFollowers = await json.business_discovery.followers_count;
      const getBiography = await json.business_discovery.biography;
      const getUserUsername = await json.business_discovery.username;
      const getUsersName = await json.business_discovery.name;
      const getProfilePicURL = await json.business_discovery
        .profile_picture_url;
      setProPicURL(getProfilePicURL);
      setFollowers(getFollowers);
      setBio(getBiography);
      setUsername(getUserUsername);
      setName(getUsersName);

      const mapURLs = await Promise.all(
        media.map(async (m) => { //async fucntion to map all posts
          const sentiAnalysis = async (caption) => { //Calls the flask server and runs the sentiment analysis ai on the caption of the currrent post
            if (caption == null) {
              return [2, 1]
            }
            try {
              const resArr = await fetch(`/getSenti?caption=${caption}`);
              const resJSON = await resArr.json();
              const sentiment = resJSON.pred;
              const proba = resJSON.percent;
              return [sentiment, proba]; //returns the sentiment and the percentage of how positve/negative it is
            } catch (error) {
              console.log("sentiAnalysis experienced an error - ", error);
            }
          };
          const result = await sentiAnalysis(m.caption);
          const sentiment = result[0];
          const proba = result[1];

          const topicIdentification = async (caption) => { //Calls the flask server and runs the topic identification ai on the caption of the currrent post
            if (caption == null) {
              return "No Topic"
            }
            try {
              const resTopic = await fetch(`/getTopic?caption=${caption}`);
              const resJSONTopic = await resTopic.json();
              const topic = resJSONTopic.pred;
              return topic; //returns the predicated topic by the ai
            } catch (error) {
              console.log("topicIdentification experienced an error - ", error);
            }
          };
          const topic = await topicIdentification(m.caption);
          setProfileTopics((prev) => [...prev, topic]);

          return { //handles having a carousel/album media type of instagram post
            caption: m.caption,
            type: m.media_type,
            sentiment: sentiment,
            proba: proba,
            topic: topic,
            url: m.children
              ? m.children.data.map((d) => d.media_url)
              : m.media_url,
          };
        })
      );
      setURLs(mapURLs);


      let count = 0;
      const sentiArr = mapURLs.map((i) => [i.proba, i.sentiment]); //Handles producing the sentiment analysis percentage of the whole profile, maps all percentages and sentiments in array
      for (let i = 0; i < sentiArr.length; i++) { //loops through the sentiment array of all the percentages and predicted sentiments
        if (sentiArr[i][1] == 1) { //sees if the sentiment is positive, if it is then it adds the percentage to the total percentage count
          count += parseFloat(sentiArr[i][0]);
        } else if (sentiArr[i][1] == 0) {
          count -= parseFloat(sentiArr[i][0]); //if the sentiment is negative, if it is then it minuses the percentage to the total percentage count
        }
        console.log("value - = " + sentiArr[i][0]);
        console.log("count " + count);
      }
      count = ((count / mapURLs.length) * 100).toFixed(0); //divides the count by the number of posts and * 100 to get a percentage to the nearest whole number
      setProfilePerc(count); //sets global variable for profile percentage to be the count calculated
    }
  }

  const [currPost, setCurrPost] = useState(null);

  useEffect(() => {setWordCloudData(setWordCloud(postURLs.map((t) => t.topic)))},[postURLs]) //sets the tag-cloud with the topic data

  //return outputs all the information in the homepage and the profile page including the profile info and analysis 
  return (
    <div className="App">
      <div className="topBar" />
      <h1>Instalyse...</h1>
      {postURLs.length == 0 ? (
        <>
          <img className="backImage" src="/EditedBackImage.png"></img>
          <Search />
        </> 
      ) : <> <button className="homepageButton" onClick={()=>setURLs([])}>Homepage</button> {currPost ? (
        <Analysis post={currPost} setCurrPost={setCurrPost} />
      ) : (
        <div className="profileContainer">
          <div className="profileInfoContainer">
            <div className="profilePicandTextContainer">
              <div className="profilePic">
                <img src={profilePicURL}></img>
              </div>
              <div className="profileTextContainer">
                <p className="showUsername">Username: {userUsername}</p>
                <p className="showName">Name: {usersName}</p>
                <p className="showFollowers">Followers: {userFollowers}</p>
                <p className="showBio">Biography: {userBiography}</p>
              </div>
            </div>
            <div className="profileAnalysisContainer">
              <div className="profilePercCircle">
                <CircularProgressbar
                  value={profilePerc == 0 ? 100 : profilePerc}
                  maxValue={100}
                  text={`${profilePerc == 0 ? 100 : profilePerc}% ${
                    profilePerc > 0 ? "Positive" : (profilePerc == 0 ? "Neutral" : "Negative")
                  }`}
                  styles={buildStyles({
                    pathColor: `#BC3DF8`,
                    trailColor: `#C0C0C0`,
                    textColor: `#A020F0`,
                    textSize: 12,
                  })}
                />
              </div>
              <div className="topicWordcloud">
                <TagCloud
                colorOptions={{
                  luminosity: 'dark',
                  hue: 'pink',
                }}
                minSize={20}
                maxSize={47}
                tags={wordCloudData} 
                />
              </div>
            </div>
          </div>
          <div className="postGrid">
            {postURLs
              .filter((m) => m.url)
              .map((post) => (
                <img
                  className="gridPics"
                  src={Array.isArray(post.url) ? post.url[0] : post.url}
                  onClick={() => setCurrPost(post)}
                ></img>
            ))}
          </div>
        </div>
      )}</>}
    </div>
  );
}

export default App;